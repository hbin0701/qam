"""
Unified Dense Reward Module for QAM

Supports any cube manipulation environment (single, double, triple, etc.)
by parameterizing on num_cubes and goal_positions, auto-detected from env_name.

Reward versions:
- v1: Simple cube placement count
- v2: Distance-based progress (recommended)
- v3: Detailed stage tracking (reach -> grasp -> carry -> place per cube)
- v4: Direct v2-delta (reward = v2(s') - v2(s) + gamma * success_bonus)
- v5: Potential-based v3 (reward = Φ_v3(s') - Φ_v3(s))
- v6: Pick-place style progress-delta (4 subtasks per cube) + terminal bonus
- v7: Pick-place style progress-delta (3 subtasks, no release stage)
      + placement event bonus/penalty + terminal bonus
- v8: v7 + explicit release event bonus/penalty
- v9: v8 without progress-potential shaping and without release event
      (stage penalty + terminal bonus)
- v10: v8-style reward with:
       (1) continuous reach progress at threshold,
       (2) hysteresis for reach->grasp switching,
       (3) continuous lift progress at grasp threshold
- v11: v10-style with extra lower stage:
       reach -> grasp -> transport(XY) -> lower(Z). Lower activates only
       inside XY success threshold and Z progress is normalized by the
       entry-Z at XY-threshold crossing.
- v12: v11 + wrong-XY penalty during lower stage:
       lower-stage subprogress is in [-1, 1], starts at 0 on entry,
       and becomes negative if XY distance increases from lower-stage entry.
- v13: v11 + subtask transition event shaping (no release event):
       +bonus for 1->2 and 2->3, same-magnitude penalty for 2->1 and 3->2.
- v14: v13 with subtask transition event enabled.
- v15: v12 + XY-worsening penalty during transport stage.
- v16: v15 with release event removed.
- v17: v16 + subtask transition event shaping.
- v18: v15 + latched lower-entry-Z per active cube (first stage-3 entry only).
- v20: v18 semantics with gripper-closure driven by physical left/right
       fingertip gap (meters) when available.
- v21: v18 semantics with the same physical gripper-gap and strict grasp
       checks as v20.
- v22: v21 semantics + monotonic stage/progress shaping (no reward for
       re-achieving previously reached stages/progress after regressions).
- v23: v22 + small per-step time penalty.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

# OGBench observation scaling constants
OBS_XYZ_CENTER = np.array([0.425, 0.0, 0.0])
OBS_XYZ_SCALER = 10.0

# OGBench cube success criterion from
# ogbench/manipspace/envs/cube_env.py::_compute_successes():
# np.linalg.norm(obj_pos - tar_pos) <= 0.04
CUBE_SUCCESS_THRESHOLD = 0.04

# Approximate gripper home: center of OGBench arm_sampling_bounds
# [[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]]
GRIPPER_HOME = np.array([0.425, 0.0, 0.275])
# Tuned for OGBench cube task2 datasets (double/triple):
# - gripper opening proxy is qpos[right_driver_joint] / 0.8 in [0, 1]
# - higher means more closed; successful lifted non-stacked states cluster near ~0.56
V6_REACH_THRESHOLD = 0.10
V6_GOAL_APPROACH_THRESHOLD = 0.09
V6_GRIPPER_CLOSED_THRESHOLD = 0.50
V6_GRIPPER_CLOSED_REF = 0.56
V6_GRASP_LIFT_THRESHOLD = 0.008
V6_GRASP_LIFT_TARGET = 0.02
# v10 reach hysteresis tuned from dataset contact-distance statistics.
V10_REACH_THRESHOLD_IN = 0.10
V10_REACH_THRESHOLD_OUT = 0.10
V10_GRASP_LIFT_THRESHOLD = 0.015
V10_MIN_CARRY_LIFT = 0.025
V10_LOWER_XY_THRESHOLD = 0.04
V10_XY_WORSEN_EPS = 0.002
V10_LIFT_DROP_EPS = 0.003
V10_FALLBACK_XY_AWAY_THRESHOLD = 0.03
V10_CLOSE_BAND_MIN = 0.50
V10_CLOSE_BAND_MAX = 0.60
V10_CLOSE_DECAY_SIGMA = 0.10
V10_CARRY_LIFT_MARGIN = 0.005
V10_CARRY_STABLE_STEPS = 2
V8_RELEASE_SUCCESS_BONUS = 0.5
V8_RELEASE_FAR_PENALTY = 0.25
V13_SUBTASK_TRANSITION_BONUS = 5.0
# v20 gripper-gap normalization (meters):
# close_score = clip((open_ref - gap) / (open_ref - close_ref), 0, 1)
# with open_ref = max(initial_gap, 0.1) and close_ref fixed at 0.05.
V20_GRIPPER_GAP_OPEN_FLOOR = 0.10
V20_GRIPPER_GAP_CLOSE_REF = 0.055
V20_PAD_CUBE_DIST_THRESHOLD = 0.045
V20_PAD_CUBE_AVG_DIST_THRESHOLD = 0.04
V21_REACH_PAD_CUBE_DIST_THRESHOLD = 0.07
V23_STEP_PENALTY = 1.0


def compute_cube_order(init_positions: np.ndarray, reference: np.ndarray = GRIPPER_HOME) -> List[int]:
    """Compute fixed cube ordering by distance from reference position.

    Returns a permutation of [0, ..., n-1] sorted by distance from reference
    to each cube's initial position (nearest first).
    """
    dists = [np.linalg.norm(reference - init_positions[i]) for i in range(len(init_positions))]
    return list(np.argsort(dists))


def extract_gripper_pos(obs: np.ndarray) -> np.ndarray:
    """Extract end-effector position from OGBench observation.

    obs[12:15] = (ee_pos - OBS_XYZ_CENTER) * OBS_XYZ_SCALER
    """
    return obs[12:15] / OBS_XYZ_SCALER + OBS_XYZ_CENTER


def gripper_close_from_gap(gripper_gap_m: float, open_ref: float, close_ref: float = V20_GRIPPER_GAP_CLOSE_REF) -> float:
    """Map physical finger gap in meters to a close score in [0, 1]."""
    denom = max(open_ref - close_ref, 1e-6)
    return float(np.clip((open_ref - gripper_gap_m) / denom, 0.0, 1.0))


def extract_gripper_gap_from_sim(env_unwrapped) -> Optional[float]:
    """Best-effort left/right fingertip-gap extraction (meters) from MuJoCo state."""
    model = getattr(env_unwrapped, "_model", None) or getattr(env_unwrapped, "model", None)
    data = getattr(env_unwrapped, "_data", None) or getattr(env_unwrapped, "data", None)
    if model is None or data is None:
        return None

    cache = getattr(env_unwrapped, "_qam_gripper_gap_pair", None)
    if cache is None:
        try:
            import mujoco
        except Exception:
            return None

        def _name2id(obj_type, name):
            try:
                return int(mujoco.mj_name2id(model, obj_type, name))
            except Exception:
                return -1

        # Prefer explicit site pairs if present; fall back to silicone-pad body centers.
        site_pairs = [
            ("ur5e/robotiq/left_fingertip_site", "ur5e/robotiq/right_fingertip_site"),
            ("ur5e/robotiq/left_fingertip", "ur5e/robotiq/right_fingertip"),
            ("left_fingertip_site", "right_fingertip_site"),
        ]
        for left_name, right_name in site_pairs:
            left_id = _name2id(mujoco.mjtObj.mjOBJ_SITE, left_name)
            right_id = _name2id(mujoco.mjtObj.mjOBJ_SITE, right_name)
            if left_id >= 0 and right_id >= 0:
                cache = ("site", left_id, right_id)
                break

        if cache is None:
            body_pairs = [
                ("ur5e/robotiq/left_silicone_pad", "ur5e/robotiq/right_silicone_pad"),
                ("ur5e/robotiq/left_pad", "ur5e/robotiq/right_pad"),
            ]
            for left_name, right_name in body_pairs:
                left_id = _name2id(mujoco.mjtObj.mjOBJ_BODY, left_name)
                right_id = _name2id(mujoco.mjtObj.mjOBJ_BODY, right_name)
                if left_id >= 0 and right_id >= 0:
                    cache = ("body", left_id, right_id)
                    break

        setattr(env_unwrapped, "_qam_gripper_gap_pair", cache)

    if cache is None:
        return None

    kind, left_id, right_id = cache
    try:
        if kind == "site":
            left = data.site_xpos[left_id]
            right = data.site_xpos[right_id]
        else:
            left = data.xpos[left_id]
            right = data.xpos[right_id]
        return float(np.linalg.norm(left - right))
    except Exception:
        return None


def extract_gripper_pad_positions_from_sim(env_unwrapped) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Best-effort extraction of left/right gripper pad world positions."""
    model = getattr(env_unwrapped, "_model", None) or getattr(env_unwrapped, "model", None)
    data = getattr(env_unwrapped, "_data", None) or getattr(env_unwrapped, "data", None)
    if model is None or data is None:
        return None, None

    cache = getattr(env_unwrapped, "_qam_gripper_pad_pair", None)
    if cache is None:
        try:
            import mujoco
        except Exception:
            return None, None

        def _name2id(obj_type, name):
            try:
                return int(mujoco.mj_name2id(model, obj_type, name))
            except Exception:
                return -1

        site_pairs = [
            ("ur5e/robotiq/left_fingertip_site", "ur5e/robotiq/right_fingertip_site"),
            ("ur5e/robotiq/left_fingertip", "ur5e/robotiq/right_fingertip"),
            ("left_fingertip_site", "right_fingertip_site"),
        ]
        for left_name, right_name in site_pairs:
            left_id = _name2id(mujoco.mjtObj.mjOBJ_SITE, left_name)
            right_id = _name2id(mujoco.mjtObj.mjOBJ_SITE, right_name)
            if left_id >= 0 and right_id >= 0:
                cache = ("site", left_id, right_id)
                break

        if cache is None:
            body_pairs = [
                ("ur5e/robotiq/left_silicone_pad", "ur5e/robotiq/right_silicone_pad"),
                ("ur5e/robotiq/left_pad", "ur5e/robotiq/right_pad"),
            ]
            for left_name, right_name in body_pairs:
                left_id = _name2id(mujoco.mjtObj.mjOBJ_BODY, left_name)
                right_id = _name2id(mujoco.mjtObj.mjOBJ_BODY, right_name)
                if left_id >= 0 and right_id >= 0:
                    cache = ("body", left_id, right_id)
                    break

        setattr(env_unwrapped, "_qam_gripper_pad_pair", cache)

    if cache is None:
        return None, None

    kind, left_id, right_id = cache
    try:
        if kind == "site":
            return data.site_xpos[left_id].copy(), data.site_xpos[right_id].copy()
        return data.xpos[left_id].copy(), data.xpos[right_id].copy()
    except Exception:
        return None, None


# ============================================================
# Task goal positions lookup: (env_type, task_id) -> goal_xyzs
# From ogbench/ogbench/manipspace/envs/cube_env.py set_tasks()
# ============================================================
TASK_GOALS = {
    # Single-play tasks (1 cube)
    ('single', 1): np.array([[0.425, -0.1, 0.02]]),
    ('single', 2): np.array([[0.50, 0.0, 0.02]]),
    ('single', 3): np.array([[0.35, 0.0, 0.02]]),
    ('single', 4): np.array([[0.50, 0.2, 0.02]]),
    ('single', 5): np.array([[0.50, -0.2, 0.02]]),
    # Double-play tasks (2 cubes)
    ('double', 1): np.array([[0.425, 0.0, 0.02], [0.425, 0.1, 0.02]]),
    ('double', 2): np.array([[0.35, 0.1, 0.02], [0.50, 0.1, 0.02]]),
    ('double', 3): np.array([[0.425, -0.2, 0.02], [0.425, 0.2, 0.02]]),
    ('double', 4): np.array([[0.425, 0.1, 0.02], [0.425, -0.1, 0.02]]),
    ('double', 5): np.array([[0.425, 0.0, 0.02], [0.425, 0.0, 0.06]]),
    # Triple-play tasks (3 cubes)
    ('triple', 1): np.array([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.50, 0.1, 0.02]]),
    ('triple', 2): np.array([[0.50, 0.0, 0.02], [0.50, 0.2, 0.02], [0.50, -0.2, 0.02]]),
    ('triple', 3): np.array([[0.35, -0.1, 0.02], [0.50, -0.2, 0.02], [0.50, 0.0, 0.02]]),
    ('triple', 4): np.array([[0.50, -0.1, 0.02], [0.50, 0.1, 0.02], [0.35, 0.0, 0.02]]),
    ('triple', 5): np.array([[0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.425, 0.2, 0.10]]),
}

# Task initial positions: used by V2 for distance-based progress computation.
# Without these, initial_position == current_position and progress_to_goal() is always 0.
TASK_INITS = {
    # Single-play tasks
    ('single', 1): np.array([[0.425, 0.1, 0.02]]),
    ('single', 2): np.array([[0.35, 0.0, 0.02]]),
    ('single', 3): np.array([[0.50, 0.0, 0.02]]),
    ('single', 4): np.array([[0.35, -0.2, 0.02]]),
    ('single', 5): np.array([[0.35, 0.2, 0.02]]),
    # Double-play tasks
    ('double', 1): np.array([[0.425, 0.0, 0.02], [0.425, -0.1, 0.02]]),
    ('double', 2): np.array([[0.35, -0.1, 0.02], [0.50, -0.1, 0.02]]),
    ('double', 3): np.array([[0.35, 0.0, 0.02], [0.50, 0.0, 0.02]]),
    ('double', 4): np.array([[0.425, -0.1, 0.02], [0.425, 0.1, 0.02]]),
    ('double', 5): np.array([[0.425, -0.2, 0.02], [0.425, 0.2, 0.02]]),
    # Triple-play tasks
    ('triple', 1): np.array([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.50, -0.1, 0.02]]),
    ('triple', 2): np.array([[0.35, -0.2, 0.02], [0.35, 0.0, 0.02], [0.35, 0.2, 0.02]]),
    ('triple', 3): np.array([[0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.425, 0.2, 0.10]]),
    ('triple', 4): np.array([[0.35, 0.0, 0.02], [0.50, -0.1, 0.02], [0.50, 0.1, 0.02]]),
    ('triple', 5): np.array([[0.35, -0.1, 0.02], [0.50, -0.2, 0.02], [0.50, 0.0, 0.02]]),
}

ENV_TYPE_NUM_CUBES = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'quadruple': 4,
    'octuple': 8,
}


def parse_env_name(env_name: str) -> Tuple[str, int, int]:
    """Parse env_name to extract env_type, task_id, and num_cubes.

    Examples:
        'cube-single-play-singletask-task4-v0' -> ('single', 4, 1)
        'cube-triple-play-singletask-task2-v0' -> ('triple', 2, 3)
    """
    # Split on '-' and match against known env types as whole tokens
    # to avoid 'single' matching inside 'singletask'
    splits = env_name.split('-')
    env_type = None
    for token in splits:
        if token in ENV_TYPE_NUM_CUBES:
            env_type = token
            break
    if env_type is None:
        raise ValueError(f"Cannot determine env_type from: {env_name}")

    task_match = re.search(r'task(\d+)', env_name)
    task_id = int(task_match.group(1)) if task_match else 2  # default task

    num_cubes = ENV_TYPE_NUM_CUBES[env_type]
    return env_type, task_id, num_cubes


# ============================================================
# Generic Cube state model
# ============================================================
class CubeState:
    """State for a single cube."""
    def __init__(self, position=None, goal_position=None, success_threshold=CUBE_SUCCESS_THRESHOLD):
        self.position: np.ndarray = np.array(position) if position is not None else np.zeros(3)
        self.initial_position: np.ndarray = self.position.copy()
        self.goal_position: np.ndarray = np.array(goal_position) if goal_position is not None else self.position.copy()
        self.success_threshold: float = success_threshold

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - self.goal_position))

    def is_at_goal(self, threshold=None) -> bool:
        if threshold is None:
            threshold = self.success_threshold
        return self.distance_to_goal() <= threshold

    def progress_to_goal(self, threshold=None) -> float:
        """Progress from initial position to goal (0.0 to 1.0), saturating at success threshold."""
        if threshold is None:
            threshold = self.success_threshold
        initial_dist = np.linalg.norm(self.initial_position - self.goal_position)
        current_dist = self.distance_to_goal()
        if initial_dist < 1e-6:
            return 1.0

        # Match environment success semantics: once within threshold, progress is complete.
        effective_current = max(current_dist - threshold, 0.0)
        effective_initial = max(initial_dist - threshold, 1e-6)
        return float(np.clip(1.0 - (effective_current / effective_initial), 0.0, 1.0))


class CubeEnvModel:
    """Lightweight physics-free environment model for reward computation."""
    def __init__(self, num_cubes: int, goal_positions: np.ndarray,
                 init_positions: Optional[np.ndarray] = None,
                 cube_order: Optional[List[int]] = None,
                 success_threshold: float = CUBE_SUCCESS_THRESHOLD):
        self.num_cubes = num_cubes
        self.goal_positions = goal_positions
        self.success_threshold = success_threshold
        self.cubes: List[CubeState] = [
            CubeState(goal_position=goal_positions[i], success_threshold=success_threshold) for i in range(num_cubes)
        ]
        # Set task initial positions for V2 distance-based progress
        if init_positions is not None:
            for i in range(num_cubes):
                self.cubes[i].initial_position = init_positions[i].copy()
        # Fixed cube ordering for v3/v5 sequential progress (nearest-first from gripper home)
        self.cube_order: List[int] = cube_order if cube_order is not None else list(range(num_cubes))
        # OGBench proxy in [0, 1]: 0=open, ~0.56=closed in these datasets.
        self.gripper_width: float = 0.0
        self.gripper_gap_m: Optional[float] = None
        self.gripper_gap_open_ref_m: float = V20_GRIPPER_GAP_OPEN_FLOOR
        self.left_gripper_pos: Optional[np.ndarray] = None
        self.right_gripper_pos: Optional[np.ndarray] = None
        self.gripper_pos: Optional[np.ndarray] = None
        self.use_pad_reach: bool = False
        self.pad_reach_threshold: float = V21_REACH_PAD_CUBE_DIST_THRESHOLD

    def load_state(self, qpos: np.ndarray, qvel: np.ndarray = None,
                   gripper_pos: Optional[np.ndarray] = None,
                   gripper_gap_m: Optional[float] = None,
                   gripper_gap_open_ref_m: Optional[float] = None,
                   left_gripper_pos: Optional[np.ndarray] = None,
                   right_gripper_pos: Optional[np.ndarray] = None):
        """Load state from qpos vector.

        qpos layout: [arm_joints(6), gripper_joints(8), cube0_pos(3)+quat(4), cube1_pos(3)+quat(4), ...]
        gripper_pos: end-effector position (from observation), needed for v3/v5 reach substage.
        """
        for i in range(self.num_cubes):
            pos_start = 14 + i * 7
            self.cubes[i].position = qpos[pos_start:pos_start + 3].copy()
        self.gripper_gap_m = None if gripper_gap_m is None else float(gripper_gap_m)
        if gripper_gap_m is not None:
            if gripper_gap_open_ref_m is None:
                gripper_gap_open_ref_m = V20_GRIPPER_GAP_OPEN_FLOOR
            self.gripper_gap_open_ref_m = float(max(gripper_gap_open_ref_m, V20_GRIPPER_GAP_OPEN_FLOOR))
            self.gripper_width = gripper_close_from_gap(float(gripper_gap_m), open_ref=self.gripper_gap_open_ref_m)
        else:
            self.gripper_width = float(np.clip(qpos[6] / 0.8, 0.0, 1.0))
            self.gripper_gap_open_ref_m = V20_GRIPPER_GAP_OPEN_FLOOR
        if gripper_pos is not None:
            self.gripper_pos = gripper_pos.copy()
        self.left_gripper_pos = left_gripper_pos.copy() if left_gripper_pos is not None else None
        self.right_gripper_pos = right_gripper_pos.copy() if right_gripper_pos is not None else None

    def is_grasped(self, cube_idx: int) -> bool:
        cube = self.cubes[cube_idx]
        gripper_closed = self.gripper_width >= V6_GRIPPER_CLOSED_THRESHOLD
        if self.gripper_gap_m is not None and self.left_gripper_pos is not None and self.right_gripper_pos is not None:
            left_dist = float(np.linalg.norm(self.left_gripper_pos - cube.position))
            right_dist = float(np.linalg.norm(self.right_gripper_pos - cube.position))
            avg_dist = 0.5 * (left_dist + right_dist)
            gripper_closed = (
                gripper_closed
                and (left_dist <= V20_PAD_CUBE_DIST_THRESHOLD)
                and (right_dist <= V20_PAD_CUBE_DIST_THRESHOLD)
                and (avg_dist <= V20_PAD_CUBE_AVG_DIST_THRESHOLD)
            )
        is_lifted = cube.position[2] > cube.initial_position[2] + 0.01
        return gripper_closed and is_lifted


# ============================================================
# V1: Simple cube placement count
# ============================================================
def progress_v1(env: CubeEnvModel) -> Tuple[float, int]:
    """Progress = number of cubes at goal. Range: [0, num_cubes]."""
    num_at_goal = sum(c.is_at_goal() for c in env.cubes)
    return (float(num_at_goal), num_at_goal)


# ============================================================
# V2: Distance-based progress
# ============================================================
def dist_based_progress(env: CubeEnvModel) -> Tuple[float, int]:
    """Progress = sum of per-cube distance-based progress.
    Each cube contributes progress_to_goal() in [0, 1].
    Range: [0, num_cubes]. Fully continuous, no gap.
    """
    total = 0.0
    completed = 0
    for cube in env.cubes:
        total += cube.progress_to_goal()
        if cube.is_at_goal():
            completed += 1
    return (total, completed)


def progress_v2(env: CubeEnvModel) -> Tuple[float, int]:
    """V2 distance-based progress."""
    return dist_based_progress(env)


def progress_v4(env: CubeEnvModel) -> Tuple[float, int]:
    """V4 uses the same distance-based progress source as V2."""
    return dist_based_progress(env)


# ============================================================
# V3: Stage tracking (not grasped -> grasped per cube)
# ============================================================
def progress_v3(env: CubeEnvModel) -> Tuple[float, int]:
    """Progress with per-cube stage tracking (reach -> place).

    For cube order position k (0-indexed):
      - Reach stage contributes [k, k + 0.5]
      - Place stage contributes [k + 0.5, k + 1.0]

    Total range: [0, num_cubes].
    """
    n = env.num_cubes

    # Find first incomplete cube in cube_order (nearest-first from gripper home)
    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        return (float(n), n)

    active_cube = env.cubes[active_idx]
    # 0: reach, 1: place
    substage = 1 if env.is_grasped(active_idx) else 0

    stage = order_pos * 2 + substage
    base_progress = float(order_pos)

    # NOTE: Per requested design, reach reference distance is computed from
    # this cube's initial position to the FIRST cube's goal position.
    first_goal = env.cubes[env.cube_order[0]].goal_position
    reach_ref_dist = float(np.linalg.norm(active_cube.initial_position - first_goal))
    reach_ref_dist = max(reach_ref_dist, 1e-6)

    place_init_dist = float(np.linalg.norm(active_cube.initial_position - active_cube.goal_position))
    current_dist = active_cube.distance_to_goal()

    # Sub-progress within current substage using fixed references
    subprogress = 0.0
    if substage == 0:
        # Reach: gripper approaching the cube.
        if env.gripper_pos is not None:
            gripper_dist = float(np.linalg.norm(env.gripper_pos - active_cube.position))
            reach_progress = float(np.clip(1.0 - gripper_dist / reach_ref_dist, 0.0, 1.0))
            subprogress = 0.5 * reach_progress
    else:
        # Place: progress moving cube from its initial location to goal.
        effective_current = max(current_dist - env.success_threshold, 0.0)
        effective_init = max(place_init_dist - env.success_threshold, 1e-6)
        place_progress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))
        subprogress = 0.5 + 0.5 * place_progress

    total = base_progress + subprogress
    return (total, stage)


def progress_v6(env: CubeEnvModel) -> Tuple[float, int]:
    """Pick-place style progress for sequential cube placement.

    For the active cube (first incomplete in fixed cube_order), progress is split
    into 4 subtasks, each worth 0.25:
      0. move-to-object
      1. grasp-object
      2. move-to-goal
      3. release-object

    Total range: [0, num_cubes].
    """
    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        return (float(n), n * 4 - 1)

    cube = env.cubes[active_idx]
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    # Open-progress proxy for release stage: 1=open, 0=closed.
    gripper_open = float(np.clip((V6_GRIPPER_CLOSED_REF - env.gripper_width) / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
    # Use v6-specific grasp condition consistent with OGBench opening proxy.
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))

    # Determine subtask index following pick-place semantics.
    if gripper_dist is not None and (gripper_dist > V6_REACH_THRESHOLD) and (not grasped):
        subtask_idx = 0  # move to object
    elif not grasped:
        subtask_idx = 1  # grasp object
    elif cube_goal_dist > V6_GOAL_APPROACH_THRESHOLD:
        subtask_idx = 2  # move to goal
    else:
        subtask_idx = 3  # release object

    # Compute normalized sub-progress for the active subtask.
    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        subprogress = 0.6 * lift_progress + 0.4 * close_progress
    elif subtask_idx == 2:
        init_goal_dist = float(np.linalg.norm(cube.initial_position - cube.goal_position))
        init_goal_dist = max(init_goal_dist, 1e-6)
        subprogress = float(np.clip(1.0 - (cube_goal_dist / init_goal_dist), 0.0, 1.0))
    else:
        subprogress = gripper_open

    main_progress = float(order_pos) + (float(subtask_idx) + float(np.clip(subprogress, 0.0, 1.0))) / 4.0
    stage = order_pos * 4 + subtask_idx
    return (main_progress, stage)


def progress_v7(env: CubeEnvModel) -> Tuple[float, int]:
    """Pick-place style progress with no release stage.

    For the active cube (first incomplete in fixed cube_order), progress is split
    into 3 subtasks:
      0. move-to-object
      1. grasp-object
      2. move-to-goal

    Total range: [0, 3 * num_cubes] in stage units.
    """
    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        return (float(n * 3), n * 3)

    cube = env.cubes[active_idx]
    cube_goal_dist = cube.distance_to_goal()
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))

    if (not grasped) and (gripper_dist is not None) and (gripper_dist > V6_REACH_THRESHOLD):
        subtask_idx = 0  # move to object
    elif not grasped:
        subtask_idx = 1  # grasp object
    else:
        subtask_idx = 2  # move to goal

    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        subprogress = 0.6 * lift_progress + 0.4 * close_progress
    else:
        init_goal_dist = float(np.linalg.norm(cube.initial_position - cube.goal_position))
        effective_current = max(cube_goal_dist - env.success_threshold, 0.0)
        effective_init = max(init_goal_dist - env.success_threshold, 1e-6)
        subprogress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))

    main_progress = float(order_pos * 3) + float(subtask_idx) + float(np.clip(subprogress, 0.0, 1.0))
    stage = order_pos * 3 + subtask_idx
    return (main_progress, stage)


def _v10_determine_current_subtask(
    env: CubeEnvModel,
    cube: CubeState,
    prev_subtask: Optional[int],
    gripper_dist: Optional[float],
) -> Tuple[int, float, float]:
    """Determine v10 subtask index and key features."""
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    # Match v8/v7 grasp condition exactly.
    grasped = (env.gripper_width >= V6_GRIPPER_CLOSED_THRESHOLD) and (cube_lift > V6_GRASP_LIFT_THRESHOLD)

    if grasped:
        return 2, cube_lift, cube_goal_xy_dist
    if gripper_dist is None:
        return 1, cube_lift, cube_goal_xy_dist

    # Match v8/v7 reach->grasp switching exactly.
    return (0 if gripper_dist > V6_REACH_THRESHOLD else 1), cube_lift, cube_goal_xy_dist


def _v10_determine_progress_for_subtask(
    env: CubeEnvModel,
    cube: CubeState,
    subtask_idx: int,
    gripper_dist: Optional[float],
    cube_lift: float,
    cube_goal_xy_dist: float,
) -> float:
    """Compute v10 subprogress (v8-equivalent shaping)."""
    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        return float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))

    if subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        return 0.6 * lift_progress + 0.4 * close_progress

    cube_goal_dist = float(np.linalg.norm(cube.position - cube.goal_position))
    init_goal_dist = float(np.linalg.norm(cube.initial_position - cube.goal_position))
    effective_current = max(cube_goal_dist - env.success_threshold, 0.0)
    effective_init = max(init_goal_dist - env.success_threshold, 1e-6)
    return float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))


def _v10_force_to_grasp(
    progress_val: float,
    stage_val: int,
    env: CubeEnvModel,
    cube: CubeState,
) -> Tuple[float, int, int]:
    """Force local stage to grasp/lift (subtask 1)."""
    order_pos_local = int(stage_val // 3)
    forced_subtask_local = 1
    forced_subprogress_local = _v10_grasp_subprogress(
        env,
        float(cube.position[2] - cube.initial_position[2]),
    )

    progress_val = float(
        order_pos_local * 3
        + float(forced_subtask_local)
        + float(np.clip(forced_subprogress_local, 0.0, 1.0))
    )
    stage_val = order_pos_local * 3 + forced_subtask_local
    return progress_val, stage_val, forced_subtask_local


def _v10_normalized_xy_distance(env: CubeEnvModel, cube: CubeState) -> float:
    init_goal_xy_dist = float(np.linalg.norm(cube.initial_position[:2] - cube.goal_position[:2]))
    effective_current_xy = max(
        float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2])) - env.success_threshold,
        0.0,
    )
    effective_init_xy = max(init_goal_xy_dist - env.success_threshold, 1e-6)
    return float(effective_current_xy / effective_init_xy)


def _v10_transport_regressed(prev_env: CubeEnvModel, curr_env: CubeEnvModel, active_idx: int) -> bool:
    prev_cube = prev_env.cubes[active_idx]
    curr_cube = curr_env.cubes[active_idx]
    prev_xy_dist = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
    curr_xy_dist = float(np.linalg.norm(curr_cube.position[:2] - curr_cube.goal_position[:2]))
    prev_lift = float(prev_cube.position[2] - prev_cube.initial_position[2])
    curr_lift = float(curr_cube.position[2] - curr_cube.initial_position[2])
    xy_worsened = curr_xy_dist > (prev_xy_dist + V10_XY_WORSEN_EPS)
    lowered_away = (
        curr_lift < (prev_lift - V10_LIFT_DROP_EPS)
        and curr_xy_dist > V10_FALLBACK_XY_AWAY_THRESHOLD
    )
    return bool(xy_worsened or lowered_away)


def _progress_v10(
    env: CubeEnvModel,
    prev_subtask: Optional[int] = None,
    prev_env: Optional[CubeEnvModel] = None,
    carry_ok: Optional[bool] = None,
) -> Tuple[float, int, int]:
    """V10 progress, implemented as a self-contained v8/v7-equivalent path."""
    _ = (prev_subtask, prev_env, carry_ok)  # Keep signature compatible with prior v10 API.

    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        done_stage = n * 3
        return (float(done_stage), done_stage, done_stage)

    cube = env.cubes[active_idx]
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))

    if (not grasped) and (gripper_dist is not None) and (gripper_dist > V6_REACH_THRESHOLD):
        subtask_idx = 0  # move to object
    elif not grasped:
        subtask_idx = 1  # grasp object
    else:
        subtask_idx = 2  # move to goal

    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        subprogress = 0.6 * lift_progress + 0.4 * close_progress
    else:
        init_goal_xy_dist = float(np.linalg.norm(cube.initial_position[:2] - cube.goal_position[:2]))
        effective_current = max(cube_goal_xy_dist - env.success_threshold, 0.0)
        effective_init = max(init_goal_xy_dist - env.success_threshold, 1e-6)
        subprogress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))

    main_progress = float(order_pos * 3) + float(subtask_idx) + float(np.clip(subprogress, 0.0, 1.0))
    stage = order_pos * 3 + subtask_idx
    return (main_progress, stage, subtask_idx)


def progress_v10(env: CubeEnvModel) -> Tuple[float, int]:
    progress, stage, _ = _progress_v10(env, prev_subtask=None)
    return progress, stage


def _progress_v11(
    env: CubeEnvModel,
    prev_subtask: Optional[int] = None,
    prev_env: Optional[CubeEnvModel] = None,
    lower_entry_z: Optional[float] = None,
    lower_entry_cube_idx: Optional[int] = None,
) -> Tuple[float, int, int, Optional[float], Optional[int]]:
    """V11 progress with explicit lower stage after XY success."""
    _ = prev_subtask
    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        done_stage = n * 4
        return (float(done_stage), done_stage, done_stage, None, None)

    if lower_entry_cube_idx != active_idx:
        lower_entry_z = None

    cube = env.cubes[active_idx]
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))
    reached_for_grasp = _is_reached_for_grasp(env, active_idx, gripper_dist)

    if (not grasped) and (not reached_for_grasp):
        subtask_idx = 0
    elif not grasped:
        subtask_idx = 1
    elif cube_goal_xy_dist > env.success_threshold:
        subtask_idx = 2
    else:
        subtask_idx = 3

    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
        lower_entry_z = None
        lower_entry_cube_idx = None
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        subprogress = 0.6 * lift_progress + 0.4 * close_progress
        lower_entry_z = None
        lower_entry_cube_idx = None
    elif subtask_idx == 2:
        init_goal_xy_dist = float(np.linalg.norm(cube.initial_position[:2] - cube.goal_position[:2]))
        effective_current = max(cube_goal_xy_dist - env.success_threshold, 0.0)
        effective_init = max(init_goal_xy_dist - env.success_threshold, 1e-6)
        subprogress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))
        lower_entry_z = None
        lower_entry_cube_idx = None
    else:
        if lower_entry_z is None:
            if prev_env is not None:
                prev_cube = prev_env.cubes[active_idx]
                prev_xy = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
                if prev_xy > prev_env.success_threshold:
                    lower_entry_z = float(prev_cube.position[2])
                else:
                    lower_entry_z = float(cube.position[2])
            else:
                lower_entry_z = float(cube.position[2])
        lower_entry_cube_idx = active_idx

        goal_z = float(cube.goal_position[2])
        entry_above = max(float(lower_entry_z - goal_z), 1e-6)
        current_above = max(float(cube.position[2] - goal_z), 0.0)
        subprogress = float(np.clip(1.0 - (current_above / entry_above), 0.0, 1.0))

    main_progress = float(order_pos * 4) + float(subtask_idx) + float(np.clip(subprogress, 0.0, 1.0))
    stage = order_pos * 4 + subtask_idx
    return (main_progress, stage, subtask_idx, lower_entry_z, lower_entry_cube_idx)


def progress_v11(env: CubeEnvModel) -> Tuple[float, int]:
    progress, stage, _, _, _ = _progress_v11(env, prev_subtask=None)
    return progress, stage


def _progress_v12(
    env: CubeEnvModel,
    prev_subtask: Optional[int] = None,
    prev_env: Optional[CubeEnvModel] = None,
    lower_entry_z: Optional[float] = None,
    lower_entry_cube_idx: Optional[int] = None,
    lower_entry_xy_dist: Optional[float] = None,
) -> Tuple[float, int, int, Optional[float], Optional[int], Optional[float]]:
    """V12 progress: v11 with lower-stage XY-worsening penalty."""
    _ = prev_subtask
    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        done_stage = n * 4
        return (float(done_stage), done_stage, done_stage, None, None, None)

    if lower_entry_cube_idx != active_idx:
        lower_entry_z = None
        lower_entry_xy_dist = None

    cube = env.cubes[active_idx]
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))
    reached_for_grasp = _is_reached_for_grasp(env, active_idx, gripper_dist)

    if (not grasped) and (not reached_for_grasp):
        subtask_idx = 0
    elif not grasped:
        subtask_idx = 1
    elif cube_goal_xy_dist > env.success_threshold:
        subtask_idx = 2
    else:
        subtask_idx = 3

    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
        lower_entry_z = None
        lower_entry_cube_idx = None
        lower_entry_xy_dist = None
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        subprogress = 0.6 * lift_progress + 0.4 * close_progress
        lower_entry_z = None
        lower_entry_cube_idx = None
        lower_entry_xy_dist = None
    elif subtask_idx == 2:
        init_goal_xy_dist = float(np.linalg.norm(cube.initial_position[:2] - cube.goal_position[:2]))
        effective_current = max(cube_goal_xy_dist - env.success_threshold, 0.0)
        effective_init = max(init_goal_xy_dist - env.success_threshold, 1e-6)
        subprogress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))
        lower_entry_z = None
        lower_entry_cube_idx = None
        lower_entry_xy_dist = None
    else:
        if lower_entry_z is None:
            if prev_env is not None:
                prev_cube = prev_env.cubes[active_idx]
                prev_xy = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
                if prev_xy > prev_env.success_threshold:
                    lower_entry_z = float(prev_cube.position[2])
                else:
                    lower_entry_z = float(cube.position[2])
            else:
                lower_entry_z = float(cube.position[2])
        if lower_entry_xy_dist is None:
            # Entry point for lower stage starts from 0 subprogress.
            lower_entry_xy_dist = cube_goal_xy_dist
        lower_entry_cube_idx = active_idx

        goal_z = float(cube.goal_position[2])
        entry_above = max(float(lower_entry_z - goal_z), 1e-6)
        current_above = max(float(cube.position[2] - goal_z), 0.0)
        z_progress = float(np.clip(1.0 - (current_above / entry_above), 0.0, 1.0))

        xy_increase = max(cube_goal_xy_dist - float(lower_entry_xy_dist), 0.0)
        xy_penalty = float(xy_increase / max(env.success_threshold, 1e-6))
        subprogress = float(np.clip(z_progress - xy_penalty, -1.0, 1.0))

    main_progress = float(order_pos * 4) + float(subtask_idx) + float(np.clip(subprogress, -1.0, 1.0))
    stage = order_pos * 4 + subtask_idx
    return (main_progress, stage, subtask_idx, lower_entry_z, lower_entry_cube_idx, lower_entry_xy_dist)


def progress_v12(env: CubeEnvModel) -> Tuple[float, int]:
    progress, stage, _, _, _, _ = _progress_v12(env, prev_subtask=None)
    return progress, stage


def _progress_v15(
    env: CubeEnvModel,
    prev_subtask: Optional[int] = None,
    prev_env: Optional[CubeEnvModel] = None,
    lower_entry_z: Optional[float] = None,
    lower_entry_cube_idx: Optional[int] = None,
    lower_entry_xy_dist: Optional[float] = None,
) -> Tuple[float, int, int, Optional[float], Optional[int], Optional[float]]:
    """V15 progress: v12 + transport-stage XY-worsening penalty (no lower-stage XY penalty)."""
    _ = prev_subtask
    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        done_stage = n * 4
        return (float(done_stage), done_stage, done_stage, None, None, None)

    if lower_entry_cube_idx != active_idx:
        lower_entry_z = None
        lower_entry_xy_dist = None

    cube = env.cubes[active_idx]
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))

    if (not grasped) and (gripper_dist is not None) and (gripper_dist > V6_REACH_THRESHOLD):
        subtask_idx = 0
    elif not grasped:
        subtask_idx = 1
    elif cube_goal_xy_dist > env.success_threshold:
        subtask_idx = 2
    else:
        subtask_idx = 3

    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
        lower_entry_z = None
        lower_entry_cube_idx = None
        lower_entry_xy_dist = None
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
        subprogress = 0.6 * lift_progress + 0.4 * close_progress
        lower_entry_z = None
        lower_entry_cube_idx = None
        lower_entry_xy_dist = None
    elif subtask_idx == 2:
        init_goal_xy_dist = float(np.linalg.norm(cube.initial_position[:2] - cube.goal_position[:2]))
        effective_current = max(cube_goal_xy_dist - env.success_threshold, 0.0)
        effective_init = max(init_goal_xy_dist - env.success_threshold, 1e-6)
        transport_progress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))
        xy_worsen_penalty = 0.0
        if prev_env is not None:
            prev_cube = prev_env.cubes[active_idx]
            prev_xy_dist = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
            xy_increase = max(cube_goal_xy_dist - prev_xy_dist, 0.0)
            xy_worsen_penalty = float(xy_increase / effective_init)
        subprogress = float(np.clip(transport_progress - xy_worsen_penalty, -1.0, 1.0))
        lower_entry_z = None
        lower_entry_cube_idx = None
        lower_entry_xy_dist = None
    else:
        if lower_entry_z is None:
            if prev_env is not None:
                prev_cube = prev_env.cubes[active_idx]
                prev_xy = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
                if prev_xy > prev_env.success_threshold:
                    lower_entry_z = float(prev_cube.position[2])
                else:
                    lower_entry_z = float(cube.position[2])
            else:
                lower_entry_z = float(cube.position[2])
        if lower_entry_xy_dist is None:
            lower_entry_xy_dist = cube_goal_xy_dist
        lower_entry_cube_idx = active_idx

        goal_z = float(cube.goal_position[2])
        entry_above = max(float(lower_entry_z - goal_z), 1e-6)
        current_above = max(float(cube.position[2] - goal_z), 0.0)
        subprogress = float(np.clip(1.0 - (current_above / entry_above), 0.0, 1.0))

    main_progress = float(order_pos * 4) + float(subtask_idx) + float(np.clip(subprogress, -1.0, 1.0))
    stage = order_pos * 4 + subtask_idx
    return (main_progress, stage, subtask_idx, lower_entry_z, lower_entry_cube_idx, lower_entry_xy_dist)


def progress_v15(env: CubeEnvModel) -> Tuple[float, int]:
    progress, stage, _, _, _, _ = _progress_v15(env, prev_subtask=None)
    return progress, stage


def progress_v16(env: CubeEnvModel) -> Tuple[float, int]:
    """V16 shares the same progress definition as v15."""
    return progress_v15(env)


def progress_v17(env: CubeEnvModel) -> Tuple[float, int]:
    """V17 shares the same progress definition as v15."""
    return progress_v15(env)


def _progress_v18(
    env: CubeEnvModel,
    prev_subtask: Optional[int] = None,
    prev_env: Optional[CubeEnvModel] = None,
    lower_entry_z: Optional[float] = None,
    lower_entry_cube_idx: Optional[int] = None,
    lower_entry_xy_dist: Optional[float] = None,
) -> Tuple[float, int, int, Optional[float], Optional[int], Optional[float]]:
    """V18 progress: v15 with latched lower-entry-Z per active cube."""
    _ = prev_subtask
    n = env.num_cubes

    active_idx = None
    order_pos = 0
    for pos, idx in enumerate(env.cube_order):
        if not env.cubes[idx].is_at_goal():
            active_idx = idx
            order_pos = pos
            break

    if active_idx is None:
        done_stage = n * 4
        return (float(done_stage), done_stage, done_stage, None, None, None)

    # Reset latch only when active cube changes.
    if lower_entry_cube_idx != active_idx:
        lower_entry_z = None
        lower_entry_xy_dist = None

    cube = env.cubes[active_idx]
    cube_goal_xy_dist = float(np.linalg.norm(cube.position[:2] - cube.goal_position[:2]))
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    grasped = _is_grasped_strict(env, active_idx, cube_lift)

    gripper_dist = None
    if env.gripper_pos is not None:
        gripper_dist = float(np.linalg.norm(env.gripper_pos - cube.position))
    reached_for_grasp = _is_reached_for_grasp(env, active_idx, gripper_dist)

    if (not grasped) and (not reached_for_grasp):
        subtask_idx = 0
    elif not grasped:
        subtask_idx = 1
    elif cube_goal_xy_dist > env.success_threshold:
        subtask_idx = 2
    else:
        subtask_idx = 3

    if subtask_idx == 0:
        init_reach_dist = float(np.linalg.norm(cube.initial_position - GRIPPER_HOME))
        init_reach_dist = max(init_reach_dist, 1e-6)
        cur_reach_dist = gripper_dist if gripper_dist is not None else init_reach_dist
        subprogress = float(np.clip(1.0 - (cur_reach_dist / init_reach_dist), 0.0, 1.0))
    elif subtask_idx == 1:
        lift_progress = float(np.clip(cube_lift / V6_GRASP_LIFT_TARGET, 0.0, 1.0))
        if env.use_pad_reach:
            # v21: remove gripper-closure component from grasp-stage subprogress.
            subprogress = lift_progress
        else:
            close_progress = float(np.clip(env.gripper_width / V6_GRIPPER_CLOSED_REF, 0.0, 1.0))
            subprogress = 0.6 * lift_progress + 0.4 * close_progress
    elif subtask_idx == 2:
        init_goal_xy_dist = float(np.linalg.norm(cube.initial_position[:2] - cube.goal_position[:2]))
        effective_current = max(cube_goal_xy_dist - env.success_threshold, 0.0)
        effective_init = max(init_goal_xy_dist - env.success_threshold, 1e-6)
        transport_progress = float(np.clip(1.0 - (effective_current / effective_init), 0.0, 1.0))
        xy_worsen_penalty = 0.0
        if prev_env is not None:
            prev_cube = prev_env.cubes[active_idx]
            prev_xy_dist = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
            xy_increase = max(cube_goal_xy_dist - prev_xy_dist, 0.0)
            xy_worsen_penalty = float(xy_increase / effective_init)
        subprogress = float(np.clip(transport_progress - xy_worsen_penalty, -1.0, 1.0))
    else:
        # Latch only first time entering lower stage for this active cube.
        if lower_entry_z is None:
            if prev_env is not None:
                prev_cube = prev_env.cubes[active_idx]
                prev_xy = float(np.linalg.norm(prev_cube.position[:2] - prev_cube.goal_position[:2]))
                if prev_xy > prev_env.success_threshold:
                    lower_entry_z = float(prev_cube.position[2])
                else:
                    lower_entry_z = float(cube.position[2])
            else:
                lower_entry_z = float(cube.position[2])
            lower_entry_xy_dist = cube_goal_xy_dist
            lower_entry_cube_idx = active_idx

        goal_z = float(cube.goal_position[2])
        entry_above = max(float(lower_entry_z - goal_z), 1e-6)
        current_above = max(float(cube.position[2] - goal_z), 0.0)
        subprogress = float(np.clip(1.0 - (current_above / entry_above), 0.0, 1.0))

    main_progress = float(order_pos * 4) + float(subtask_idx) + float(np.clip(subprogress, -1.0, 1.0))
    stage = order_pos * 4 + subtask_idx
    return (main_progress, stage, subtask_idx, lower_entry_z, lower_entry_cube_idx, lower_entry_xy_dist)


def progress_v18(env: CubeEnvModel) -> Tuple[float, int]:
    progress, stage, _, _, _, _ = _progress_v18(env, prev_subtask=None)
    return progress, stage


def progress_v13(env: CubeEnvModel) -> Tuple[float, int]:
    """V13 shares the same progress definition as v11."""
    return progress_v11(env)


def progress_v14(env: CubeEnvModel) -> Tuple[float, int]:
    """V14 shares the same progress definition as v11."""
    return progress_v11(env)


def _v10_grasp_subprogress(env: CubeEnvModel, cube_lift: float) -> float:
    """Subprogress for v10 grasp/lift stage."""
    lift_denom = max(V10_MIN_CARRY_LIFT - V10_GRASP_LIFT_THRESHOLD, 1e-6)
    lift_eff = max(cube_lift - V10_GRASP_LIFT_THRESHOLD, 0.0)
    lift_progress = float(np.clip(lift_eff / lift_denom, 0.0, 1.0))

    # Close-progress profile:
    # - inside [0.5, 0.6] -> 1.0
    # - outside band -> exponential decay by distance to band
    gw = float(env.gripper_width)
    d = max(V10_CLOSE_BAND_MIN - gw, gw - V10_CLOSE_BAND_MAX, 0.0)
    sigma = max(V10_CLOSE_DECAY_SIGMA, 1e-6)
    close_progress = float(np.exp(-((d / sigma) ** 2)))

    return 0.6 * lift_progress + 0.4 * close_progress


def _is_grasped_strict(env: CubeEnvModel, cube_idx: int, cube_lift: float) -> bool:
    """Base grasp test with stricter v20 pad-to-cube distance checks."""
    lift_threshold = V6_GRASP_LIFT_TARGET if env.use_pad_reach else V6_GRASP_LIFT_THRESHOLD
    grasped = (env.gripper_width >= V6_GRIPPER_CLOSED_THRESHOLD) and (cube_lift > lift_threshold)
    if not grasped:
        return False
    if env.gripper_gap_m is None:
        return True
    if env.left_gripper_pos is None or env.right_gripper_pos is None:
        return True
    cube = env.cubes[cube_idx]
    left_dist = float(np.linalg.norm(env.left_gripper_pos - cube.position))
    right_dist = float(np.linalg.norm(env.right_gripper_pos - cube.position))
    avg_dist = 0.5 * (left_dist + right_dist)
    return (
        (left_dist <= V20_PAD_CUBE_DIST_THRESHOLD)
        and (right_dist <= V20_PAD_CUBE_DIST_THRESHOLD)
        and (avg_dist <= V20_PAD_CUBE_AVG_DIST_THRESHOLD)
    )


def _is_reached_for_grasp(env: CubeEnvModel, cube_idx: int, gripper_dist: Optional[float]) -> bool:
    """Reach condition: for v21 use bilateral pad-to-cube thresholds, else fall back."""
    if env.use_pad_reach:
        if env.left_gripper_pos is None or env.right_gripper_pos is None:
            return False
        cube = env.cubes[cube_idx]
        left_dist = float(np.linalg.norm(env.left_gripper_pos - cube.position))
        right_dist = float(np.linalg.norm(env.right_gripper_pos - cube.position))
        return (left_dist <= env.pad_reach_threshold) and (right_dist <= env.pad_reach_threshold)
    if gripper_dist is None:
        return False
    return gripper_dist <= V6_REACH_THRESHOLD


# ============================================================
# Unified DenseRewardWrapper
# ============================================================
PROGRESS_FNS = {
    'v1': progress_v1,
    'v2': progress_v2,
    'v3': progress_v3,
    'v4': progress_v4,  # Potential-based version of v2
    'v5': progress_v3,  # Potential-based version of v3
    'v6': progress_v6,  # Pick-place style delta shaping
    'v7': progress_v7,  # Pick-place without release stage
    'v8': progress_v7,  # v7 progress + explicit release event shaping
    'v9': progress_v7,  # v8 structure without progress-potential shaping/event
    'v10': progress_v10,
    'v11': progress_v11,
    'v12': progress_v12,
    'v15': progress_v15,
    'v16': progress_v16,
    'v17': progress_v17,
    'v18': progress_v18,
    'v20': progress_v18,
    'v21': progress_v18,
    'v22': progress_v18,
    'v23': progress_v18,
    'v13': progress_v13,
    'v14': progress_v14,
}


class DenseRewardWrapper:
    """Unified wrapper for dense reward computation.

    Automatically detects num_cubes and goal_positions from env_name.

    Usage:
        wrapper = DenseRewardWrapper(
            task_name="cube-single-play-singletask-task4-v0",
            version="v2",
        )
        rewards = wrapper.compute_dataset_rewards(ds, discount=0.99, terminal_bonus=1.0)
    """

    def __init__(
        self,
        task_name: str,
        version: str = "v2",
        debug: bool = False,
        success_threshold: float = CUBE_SUCCESS_THRESHOLD,
        v23_step_penalty: float = V23_STEP_PENALTY,
    ):
        self.task_name = task_name
        self.version = version
        self.debug = debug
        self.success_threshold = success_threshold
        self.v23_step_penalty = float(v23_step_penalty)

        # Auto-detect from env_name
        self.env_type, self.task_id, self.num_cubes = parse_env_name(task_name)

        key = (self.env_type, self.task_id)
        if key not in TASK_GOALS:
            raise ValueError(
                f"No goal positions for ({self.env_type}, task{self.task_id}). "
                f"Available: {list(TASK_GOALS.keys())}"
            )
        self.goal_positions = TASK_GOALS[key]
        self.init_positions = TASK_INITS.get(key)
        # Per-episode init positions (set from actual reset state for online/eval).
        # When set, these override static TASK_INITS.
        self.episode_init_positions: Optional[np.ndarray] = None
        self._v10_prev_subtask: Optional[int] = None
        self._v10_carry_stable_steps: int = 0
        self._v11_prev_subtask: Optional[int] = None
        self._v11_lower_entry_z: Optional[float] = None
        self._v11_lower_entry_cube_idx: Optional[int] = None
        self._v12_prev_subtask: Optional[int] = None
        self._v12_lower_entry_z: Optional[float] = None
        self._v12_lower_entry_cube_idx: Optional[int] = None
        self._v12_lower_entry_xy_dist: Optional[float] = None
        self._v13_prev_subtask: Optional[int] = None
        self._v13_lower_entry_z: Optional[float] = None
        self._v13_lower_entry_cube_idx: Optional[int] = None
        self._v14_prev_subtask: Optional[int] = None
        self._v14_lower_entry_z: Optional[float] = None
        self._v14_lower_entry_cube_idx: Optional[int] = None
        self._v15_prev_subtask: Optional[int] = None
        self._v15_lower_entry_z: Optional[float] = None
        self._v15_lower_entry_cube_idx: Optional[int] = None
        self._v15_lower_entry_xy_dist: Optional[float] = None
        self._v16_prev_subtask: Optional[int] = None
        self._v16_lower_entry_z: Optional[float] = None
        self._v16_lower_entry_cube_idx: Optional[int] = None
        self._v16_lower_entry_xy_dist: Optional[float] = None
        self._v17_prev_subtask: Optional[int] = None
        self._v17_lower_entry_z: Optional[float] = None
        self._v17_lower_entry_cube_idx: Optional[int] = None
        self._v17_lower_entry_xy_dist: Optional[float] = None
        self._v18_prev_subtask: Optional[int] = None
        self._v18_lower_entry_z: Optional[float] = None
        self._v18_lower_entry_cube_idx: Optional[int] = None
        self._v18_lower_entry_xy_dist: Optional[float] = None
        self._v20_prev_subtask: Optional[int] = None
        self._v20_lower_entry_z: Optional[float] = None
        self._v20_lower_entry_cube_idx: Optional[int] = None
        self._v20_lower_entry_xy_dist: Optional[float] = None
        self._v20_gap_open_ref_m: float = V20_GRIPPER_GAP_OPEN_FLOOR
        self._v22_prev_subtask: Optional[int] = None
        self._v22_lower_entry_z: Optional[float] = None
        self._v22_lower_entry_cube_idx: Optional[int] = None
        self._v22_lower_entry_xy_dist: Optional[float] = None
        self._v22_gap_open_ref_m: float = V20_GRIPPER_GAP_OPEN_FLOOR
        self._v22_best_progress: Optional[float] = None
        self._v22_best_stage: Optional[int] = None

        # Precompute cube ordering for v3/v5/v6: nearest cube to gripper home goes first
        if self.init_positions is not None:
            self.cube_order = compute_cube_order(self.init_positions)
        else:
            self.cube_order = list(range(self.num_cubes))

        if version not in ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v20', 'v21', 'v22', 'v23'):
            raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v20', 'v21', 'v22', 'v23'")

        if debug:
            print(f"[DenseReward-{version}] env_type={self.env_type}, task_id={self.task_id}, "
                  f"num_cubes={self.num_cubes}")
            print(f"[DenseReward-{version}] goal_positions={self.goal_positions.tolist()}")
            print(f"[DenseReward-{version}] cube_order={self.cube_order}")

    @property
    def max_progress(self) -> float:
        """Maximum raw progress value for this version."""
        if self.version in ('v3', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v20', 'v21', 'v22', 'v23'):
            return float(self.num_cubes)
        return float(self.num_cubes)

    def compute_potential(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None,
                          gripper_pos: Optional[np.ndarray] = None,
                          gripper_gap_m: Optional[float] = None,
                          gripper_gap_open_ref_m: Optional[float] = None,
                          left_gripper_pos: Optional[np.ndarray] = None,
                          right_gripper_pos: Optional[np.ndarray] = None) -> float:
        """Compute shaping potential/value from progress.

        - v2/v3 use raw progress P_v2(s)/P_v3(s), range [0, num_cubes].
        - other versions keep centered form, range [-num_cubes, 0].
        """
        progress, _ = self.compute_progress(
            qpos,
            qvel,
            gripper_pos,
            gripper_gap_m=gripper_gap_m,
            gripper_gap_open_ref_m=gripper_gap_open_ref_m,
            left_gripper_pos=left_gripper_pos,
            right_gripper_pos=right_gripper_pos,
        )
        if self.version in ('v2', 'v3', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v20', 'v21', 'v22', 'v23'):
            return float(progress)
        scale = self.num_cubes / self.max_progress
        return float(progress * scale - self.num_cubes)

    def _make_env(self) -> CubeEnvModel:
        init_positions = self.episode_init_positions if self.episode_init_positions is not None else self.init_positions
        return CubeEnvModel(
            self.num_cubes,
            self.goal_positions,
            init_positions,
            self.cube_order,
            success_threshold=self.success_threshold,
        )

    def set_episode_initial_positions_from_qpos(self, qpos: np.ndarray) -> None:
        """Capture cube initial positions from the actual episode reset state."""
        init_positions = []
        for i in range(self.num_cubes):
            pos_start = 14 + i * 7
            init_positions.append(qpos[pos_start:pos_start + 3].copy())
        self.episode_init_positions = np.array(init_positions, dtype=np.float64)
        self._v10_prev_subtask = None
        self._v10_carry_stable_steps = 0
        self._v11_prev_subtask = None
        self._v11_lower_entry_z = None
        self._v11_lower_entry_cube_idx = None
        self._v12_prev_subtask = None
        self._v12_lower_entry_z = None
        self._v12_lower_entry_cube_idx = None
        self._v12_lower_entry_xy_dist = None
        self._v13_prev_subtask = None
        self._v13_lower_entry_z = None
        self._v13_lower_entry_cube_idx = None
        self._v14_prev_subtask = None
        self._v14_lower_entry_z = None
        self._v14_lower_entry_cube_idx = None
        self._v15_prev_subtask = None
        self._v15_lower_entry_z = None
        self._v15_lower_entry_cube_idx = None
        self._v15_lower_entry_xy_dist = None
        self._v16_prev_subtask = None
        self._v16_lower_entry_z = None
        self._v16_lower_entry_cube_idx = None
        self._v16_lower_entry_xy_dist = None
        self._v17_prev_subtask = None
        self._v17_lower_entry_z = None
        self._v17_lower_entry_cube_idx = None
        self._v17_lower_entry_xy_dist = None
        self._v18_prev_subtask = None
        self._v18_lower_entry_z = None
        self._v18_lower_entry_cube_idx = None
        self._v18_lower_entry_xy_dist = None
        self._v20_prev_subtask = None
        self._v20_lower_entry_z = None
        self._v20_lower_entry_cube_idx = None
        self._v20_lower_entry_xy_dist = None
        self._v20_gap_open_ref_m = V20_GRIPPER_GAP_OPEN_FLOOR
        self._v22_prev_subtask = None
        self._v22_lower_entry_z = None
        self._v22_lower_entry_cube_idx = None
        self._v22_lower_entry_xy_dist = None
        self._v22_gap_open_ref_m = V20_GRIPPER_GAP_OPEN_FLOOR
        self._v22_best_progress = None
        self._v22_best_stage = None

    @staticmethod
    def _monotonic_progress_stage(
        prev_progress: float,
        curr_progress: float,
        prev_stage: int,
        curr_stage: int,
        best_progress: Optional[float],
        best_stage: Optional[int],
    ) -> Tuple[float, float, int, float, int]:
        """Compute monotonicized progress/stage for anti-cycling shaping."""
        effective_prev_progress = max(float(prev_progress), float(best_progress) if best_progress is not None else float(prev_progress))
        effective_curr_progress = max(effective_prev_progress, float(curr_progress))
        effective_prev_stage = max(int(prev_stage), int(best_stage) if best_stage is not None else int(prev_stage))
        effective_curr_stage = max(effective_prev_stage, int(curr_stage))
        return (
            effective_prev_progress,
            effective_curr_progress,
            effective_curr_stage,
            effective_curr_progress,
            effective_curr_stage,
        )

    @staticmethod
    def _stage_penalty_from_stage(stage: int, num_cubes: int, num_substages: int = 3) -> float:
        total_stages = num_cubes * num_substages
        if stage >= total_stages:
            return 0.0
        order_pos = int(stage // num_substages)
        subtask_idx = int(stage % num_substages)
        num_incomplete = num_cubes - order_pos
        remaining_full_substages = (num_incomplete - 1) * num_substages
        current_subtask_penalty = num_substages - subtask_idx
        return float(-(remaining_full_substages + current_subtask_penalty))

    def _compute_v10_progress_with_state(
        self,
        qpos: np.ndarray,
        qvel: Optional[np.ndarray] = None,
        ob: Optional[np.ndarray] = None,
        prev_subtask: Optional[int] = None,
        prev_qpos: Optional[np.ndarray] = None,
        prev_qvel: Optional[np.ndarray] = None,
        prev_ob: Optional[np.ndarray] = None,
        use_stateful_carry_counter: bool = False,
    ) -> Tuple[float, int, int]:
        def _load_env(state_qpos, state_qvel=None, state_ob=None):
            gp_local = extract_gripper_pos(state_ob) if state_ob is not None else None
            env_local = self._make_env()
            env_local.load_state(state_qpos, qvel=state_qvel, gripper_pos=gp_local)
            return env_local

        curr_env = _load_env(qpos, qvel, ob)
        prev_env = _load_env(prev_qpos, prev_qvel, prev_ob) if prev_qpos is not None else None

        return _progress_v10(
            curr_env,
            prev_subtask=prev_subtask,
            prev_env=prev_env, 
            carry_ok=None,
        )

    @staticmethod
    def _active_cube_idx(env: CubeEnvModel) -> Optional[int]:
        for idx in env.cube_order:
            if not env.cubes[idx].is_at_goal():
                return idx
        return None

    def _v8_release_event(self, prev_qpos: np.ndarray, curr_qpos: np.ndarray, prev_ob=None, curr_ob=None) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, gripper_pos=curr_gp)

        active_idx = self._active_cube_idx(prev_env)
        if active_idx is None:
            return 0.0

        was_grasped = prev_env.is_grasped(active_idx)
        is_grasped = curr_env.is_grasped(active_idx)
        if not (was_grasped and not is_grasped):
            return 0.0

        release_dist = curr_env.cubes[active_idx].distance_to_goal()
        if release_dist <= self.success_threshold:
            return V8_RELEASE_SUCCESS_BONUS
        return -V8_RELEASE_FAR_PENALTY

    @staticmethod
    def _v13_subtask_transition_event(prev_subtask: int, next_subtask: int) -> float:
        if (prev_subtask, next_subtask) in ((1, 2), (2, 3)):
            return V13_SUBTASK_TRANSITION_BONUS
        if (prev_subtask, next_subtask) in ((2, 1), (3, 2)):
            return -V13_SUBTASK_TRANSITION_BONUS
        return 0.0

    def _v7_v8_stage_penalty(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None, ob=None) -> float:
        """Stage-dependent step penalty for v7/v8, scaled by remaining cubes."""
        gp = extract_gripper_pos(ob) if ob is not None else None
        env = self._make_env()
        env.load_state(qpos, qvel=qvel, gripper_pos=gp)
        _, stage = progress_v7(env)
        return self._stage_penalty_from_stage(stage=stage, num_cubes=env.num_cubes, num_substages=3)

    def compute_progress(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None,
                         gripper_pos: Optional[np.ndarray] = None,
                         gripper_gap_m: Optional[float] = None,
                         gripper_gap_open_ref_m: Optional[float] = None,
                         left_gripper_pos: Optional[np.ndarray] = None,
                         right_gripper_pos: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """Compute progress for a single state."""
        env = self._make_env()
        env.load_state(
            qpos,
            qvel,
            gripper_pos,
            gripper_gap_m=gripper_gap_m,
            gripper_gap_open_ref_m=gripper_gap_open_ref_m,
            left_gripper_pos=left_gripper_pos,
            right_gripper_pos=right_gripper_pos,
        )
        env.use_pad_reach = self.version in ("v21", "v22", "v23")
        env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD
        return PROGRESS_FNS[self.version](env)

    def compute_progress_for_logging(
        self,
        qpos: np.ndarray,
        ob: Optional[np.ndarray] = None,
        gripper_gap_m: Optional[float] = None,
        gripper_gap_open_ref_m: Optional[float] = None,
        left_gripper_pos: Optional[np.ndarray] = None,
        right_gripper_pos: Optional[np.ndarray] = None,
    ) -> Tuple[float, int]:
        """Progress for visualization using current internal state (for stateful versions)."""
        gp = extract_gripper_pos(ob) if ob is not None else None
        env = self._make_env()
        env.load_state(
            qpos,
            qvel=None,
            gripper_pos=gp,
            gripper_gap_m=gripper_gap_m,
            gripper_gap_open_ref_m=gripper_gap_open_ref_m,
            left_gripper_pos=left_gripper_pos,
            right_gripper_pos=right_gripper_pos,
        )
        env.use_pad_reach = self.version in ("v21", "v22", "v23")
        env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD

        if self.version == 'v11':
            p, s, _, _, _ = _progress_v11(
                env,
                prev_subtask=self._v11_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v11_lower_entry_z,
                lower_entry_cube_idx=self._v11_lower_entry_cube_idx,
            )
            return float(p), int(s)
        if self.version == 'v12':
            p, s, _, _, _, _ = _progress_v12(
                env,
                prev_subtask=self._v12_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v12_lower_entry_z,
                lower_entry_cube_idx=self._v12_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v12_lower_entry_xy_dist,
            )
            return float(p), int(s)
        if self.version == 'v13':
            p, s, _, _, _ = _progress_v11(
                env,
                prev_subtask=self._v13_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v13_lower_entry_z,
                lower_entry_cube_idx=self._v13_lower_entry_cube_idx,
            )
            return float(p), int(s)
        if self.version == 'v14':
            p, s, _, _, _ = _progress_v11(
                env,
                prev_subtask=self._v14_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v14_lower_entry_z,
                lower_entry_cube_idx=self._v14_lower_entry_cube_idx,
            )
            return float(p), int(s)
        if self.version == 'v15':
            p, s, _, _, _, _ = _progress_v15(
                env,
                prev_subtask=self._v15_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v15_lower_entry_z,
                lower_entry_cube_idx=self._v15_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v15_lower_entry_xy_dist,
            )
            return float(p), int(s)
        if self.version == 'v16':
            p, s, _, _, _, _ = _progress_v15(
                env,
                prev_subtask=self._v16_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v16_lower_entry_z,
                lower_entry_cube_idx=self._v16_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v16_lower_entry_xy_dist,
            )
            return float(p), int(s)
        if self.version == 'v17':
            p, s, _, _, _, _ = _progress_v15(
                env,
                prev_subtask=self._v17_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v17_lower_entry_z,
                lower_entry_cube_idx=self._v17_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v17_lower_entry_xy_dist,
            )
            return float(p), int(s)
        if self.version == 'v18':
            p, s, _, _, _, _ = _progress_v18(
                env,
                prev_subtask=self._v18_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v18_lower_entry_z,
                lower_entry_cube_idx=self._v18_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v18_lower_entry_xy_dist,
            )
            return float(p), int(s)
        if self.version in ('v20', 'v21'):
            p, s, _, _, _, _ = _progress_v18(
                env,
                prev_subtask=self._v20_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v20_lower_entry_z,
                lower_entry_cube_idx=self._v20_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v20_lower_entry_xy_dist,
            )
            return float(p), int(s)
        if self.version in ('v22', 'v23'):
            p, s, _, _, _, _ = _progress_v18(
                env,
                prev_subtask=self._v22_prev_subtask,
                prev_env=None,
                lower_entry_z=self._v22_lower_entry_z,
                lower_entry_cube_idx=self._v22_lower_entry_cube_idx,
                lower_entry_xy_dist=self._v22_lower_entry_xy_dist,
            )
            return float(p), int(s)

        return self.compute_progress(
            qpos,
            qvel=None,
            gripper_pos=gp,
            gripper_gap_m=gripper_gap_m,
            gripper_gap_open_ref_m=gripper_gap_open_ref_m,
            left_gripper_pos=left_gripper_pos,
            right_gripper_pos=right_gripper_pos,
        )

    def _require_keys(self, ds: Dict[str, np.ndarray], keys: List[str], ctx: str):
        missing = [k for k in keys if k not in ds]
        if missing:
            raise ValueError(f"{ctx} requires keys: {missing}")

    def _is_success_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> bool:
        """Task success from post-state only (all cubes at goal)."""
        env = self._make_env()
        env.load_state(qpos, qvel=qvel, gripper_pos=None)
        return all(c.is_at_goal() for c in env.cubes)

    def count_success_cubes(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> int:
        """Count cubes currently within success threshold."""
        env = self._make_env()
        env.load_state(qpos, qvel=qvel, gripper_pos=None)
        return int(sum(c.is_at_goal() for c in env.cubes))

    def _success_bonus_post(self, next_qpos: np.ndarray, next_qvel: Optional[np.ndarray], terminal_bonus: float) -> float:
        """Post-success bonus: bonus_t = beta * 1[success(s_{t+1})]."""
        return terminal_bonus if self._is_success_state(next_qpos, qvel=next_qvel) else 0.0

    def compute_v1_dataset_rewards(self, ds: Dict[str, np.ndarray], terminal_bonus: float = 50.0, **_) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos'], "v1")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            out[i] = self.compute_potential(qpos_data[i], qvel) + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v2_dataset_rewards(self, ds: Dict[str, np.ndarray], terminal_bonus: float = 50.0, **_) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos'], "v2")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            out[i] = self.compute_potential(qpos_data[i], qvel) + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v3_dataset_rewards(self, ds: Dict[str, np.ndarray], terminal_bonus: float = 50.0, **_) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos'], "v3")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        obs_data = ds.get('observations', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            gp = extract_gripper_pos(obs_data[i]) if obs_data is not None else None
            out[i] = self.compute_potential(qpos_data[i], qvel, gripper_pos=gp) + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v4_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v4")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        base_rewards = ds['rewards']
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel)
            shaping = shaping_coef * (discount * next_progress - curr_progress)
            out[i] = base_rewards[i] + shaping + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v5_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v5")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        base_rewards = ds['rewards']
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            gp = extract_gripper_pos(obs_data[i]) if obs_data is not None else None
            next_gp = extract_gripper_pos(next_obs_data[i]) if next_obs_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel, gripper_pos=gp)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel, gripper_pos=next_gp)
            shaping = shaping_coef * (discount * next_progress - curr_progress)
            out[i] = base_rewards[i] + shaping + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v6_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v6")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        base_rewards = ds['rewards']
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            gp = extract_gripper_pos(obs_data[i]) if obs_data is not None else None
            next_gp = extract_gripper_pos(next_obs_data[i]) if next_obs_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel, gripper_pos=gp)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel, gripper_pos=next_gp)
            shaping = shaping_coef * (discount * next_progress - curr_progress)
            out[i] = base_rewards[i] + shaping + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v7_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v7")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            gp = extract_gripper_pos(obs_data[i]) if obs_data is not None else None
            next_gp = extract_gripper_pos(next_obs_data[i]) if next_obs_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel, gripper_pos=gp)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel, gripper_pos=next_gp)
            shaping = shaping_coef * (discount * next_progress - curr_progress)
            stage_penalty = self._v7_v8_stage_penalty(
                qpos=next_qpos_data[i],
                qvel=next_qvel,
                ob=next_obs_data[i] if next_obs_data is not None else None,
            )

            out[i] = stage_penalty + shaping + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v8_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v8")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            gp = extract_gripper_pos(obs_data[i]) if obs_data is not None else None
            next_gp = extract_gripper_pos(next_obs_data[i]) if next_obs_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel, gripper_pos=gp)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel, gripper_pos=next_gp)
            shaping = shaping_coef * (discount * next_progress - curr_progress)
            stage_penalty = self._v7_v8_stage_penalty(
                qpos=next_qpos_data[i],
                qvel=next_qvel,
                ob=next_obs_data[i] if next_obs_data is not None else None,
            )
            event = self._v8_release_event(
                prev_qpos=qpos_data[i],
                curr_qpos=next_qpos_data[i],
                prev_ob=obs_data[i] if obs_data is not None else None,
                curr_ob=next_obs_data[i] if next_obs_data is not None else None,
            )
            out[i] = stage_penalty + shaping + event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v9_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v9")
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        next_obs_data = ds.get('next_observations', None)
        out = np.zeros(len(next_qpos_data), dtype=np.float32)
        for i in range(len(next_qpos_data)):
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            stage_penalty = self._v7_v8_stage_penalty(
                qpos=next_qpos_data[i],
                qvel=next_qvel,
                ob=next_obs_data[i] if next_obs_data is not None else None,
            )
            out[i] = stage_penalty + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v10_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v10")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            gp = extract_gripper_pos(obs_data[i]) if obs_data is not None else None
            next_gp = extract_gripper_pos(next_obs_data[i]) if next_obs_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel, gripper_pos=gp)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel, gripper_pos=next_gp)
            shaping = shaping_coef * (discount * next_progress - curr_progress)
            stage_penalty = self._v7_v8_stage_penalty(
                qpos=next_qpos_data[i],
                qvel=next_qvel,
                ob=next_obs_data[i] if next_obs_data is not None else None,
            )
            event = self._v8_release_event(
                prev_qpos=qpos_data[i],
                curr_qpos=next_qpos_data[i],
                prev_ob=obs_data[i] if obs_data is not None else None,
                curr_ob=next_obs_data[i] if next_obs_data is not None else None,
            )
            out[i] = stage_penalty + shaping + event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v11_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v11")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev = _progress_v11(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next = _progress_v11(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            event = self._v8_release_event(
                prev_qpos=qpos_data[i],
                curr_qpos=next_qpos_data[i],
                prev_ob=prev_ob_i,
                curr_ob=next_ob_i,
            )
            out[i] = stage_penalty + shaping + event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
        return out

    def compute_v12_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v12")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None
        lower_entry_xy_dist_state: Optional[float] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v12(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
                lower_entry_xy_dist=lower_entry_xy_dist_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next, lower_entry_xy_next = _progress_v12(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
                lower_entry_xy_dist=lower_entry_xy_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            event = self._v8_release_event(
                prev_qpos=qpos_data[i],
                curr_qpos=next_qpos_data[i],
                prev_ob=prev_ob_i,
                curr_ob=next_ob_i,
            )
            out[i] = stage_penalty + shaping + event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            lower_entry_xy_dist_state = lower_entry_xy_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
                lower_entry_xy_dist_state = None
        return out

    def compute_v13_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v13")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev = _progress_v11(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next = _progress_v11(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            # transition_event = self._v13_subtask_transition_event(
            #     prev_subtask=prev_subtask_resolved,
            #     next_subtask=next_subtask,
            # )
            out[i] = stage_penalty + shaping + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
        return out

    def compute_v15_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v15")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None
        lower_entry_xy_dist_state: Optional[float] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v15(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
                lower_entry_xy_dist=lower_entry_xy_dist_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next, lower_entry_xy_next = _progress_v15(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
                lower_entry_xy_dist=lower_entry_xy_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            event = self._v8_release_event(
                prev_qpos=qpos_data[i],
                curr_qpos=next_qpos_data[i],
                prev_ob=prev_ob_i,
                curr_ob=next_ob_i,
            )
            out[i] = stage_penalty + shaping + event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            lower_entry_xy_dist_state = lower_entry_xy_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
                lower_entry_xy_dist_state = None
        return out

    def compute_v16_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v16")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None
        lower_entry_xy_dist_state: Optional[float] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v15(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
                lower_entry_xy_dist=lower_entry_xy_dist_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next, lower_entry_xy_next = _progress_v15(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
                lower_entry_xy_dist=lower_entry_xy_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            out[i] = stage_penalty + shaping + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            lower_entry_xy_dist_state = lower_entry_xy_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
                lower_entry_xy_dist_state = None
        return out

    def compute_v17_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v17")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None
        lower_entry_xy_dist_state: Optional[float] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v15(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
                lower_entry_xy_dist=lower_entry_xy_dist_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next, lower_entry_xy_next = _progress_v15(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
                lower_entry_xy_dist=lower_entry_xy_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            transition_event = self._v13_subtask_transition_event(
                prev_subtask=prev_subtask_resolved,
                next_subtask=next_subtask,
            )
            out[i] = stage_penalty + shaping + transition_event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            lower_entry_xy_dist_state = lower_entry_xy_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
                lower_entry_xy_dist_state = None
        return out

    def compute_v18_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v18")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None
        lower_entry_xy_dist_state: Optional[float] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            prev_env.use_pad_reach = self.version in ("v21", "v22", "v23")
            prev_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)
            next_env.use_pad_reach = self.version in ("v21", "v22", "v23")
            next_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v18(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
                lower_entry_xy_dist=lower_entry_xy_dist_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next, lower_entry_xy_next = _progress_v18(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
                lower_entry_xy_dist=lower_entry_xy_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            event = self._v8_release_event(
                prev_qpos=qpos_data[i],
                curr_qpos=next_qpos_data[i],
                prev_ob=prev_ob_i,
                curr_ob=next_ob_i,
            )
            out[i] = stage_penalty + shaping + event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            lower_entry_xy_dist_state = lower_entry_xy_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
                lower_entry_xy_dist_state = None
        return out

    def compute_v20_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        """V20 reuses v18 dataset path (no MuJoCo fingertip gap in offline NPZ)."""
        return self.compute_v18_dataset_rewards(
            ds=ds,
            discount=discount,
            terminal_bonus=terminal_bonus,
            shaping_coef=shaping_coef,
        )

    def compute_v22_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        step_penalty: float = 0.0,
    ) -> np.ndarray:
        """V22/V23 dataset rewards: v21 semantics with monotonic anti-cycling."""
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v22")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None
        lower_entry_xy_dist_state: Optional[float] = None
        best_progress_state: Optional[float] = None
        best_stage_state: Optional[int] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            prev_env.use_pad_reach = self.version in ("v21", "v22", "v23")
            prev_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)
            next_env.use_pad_reach = self.version in ("v21", "v22", "v23")
            next_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD

            prev_progress, prev_stage, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v18(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
                lower_entry_xy_dist=lower_entry_xy_dist_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next, lower_entry_xy_next = _progress_v18(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
                lower_entry_xy_dist=lower_entry_xy_prev,
            )
            mono_prev_progress, mono_next_progress, mono_next_stage, best_progress_next, best_stage_next = self._monotonic_progress_stage(
                prev_progress=prev_progress,
                curr_progress=next_progress,
                prev_stage=prev_stage,
                curr_stage=next_stage,
                best_progress=best_progress_state,
                best_stage=best_stage_state,
            )

            shaping = shaping_coef * (discount * mono_next_progress - mono_prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=mono_next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            out[i] = (
                stage_penalty
                + shaping
                + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
                - float(step_penalty)
            )

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            lower_entry_xy_dist_state = lower_entry_xy_next
            best_progress_state = best_progress_next
            best_stage_state = best_stage_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
                lower_entry_xy_dist_state = None
                best_progress_state = None
                best_stage_state = None
        return out

    def compute_v23_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        return self.compute_v22_dataset_rewards(
            ds=ds,
            discount=discount,
            terminal_bonus=terminal_bonus,
            shaping_coef=shaping_coef,
            step_penalty=self.v23_step_penalty,
        )

    def compute_v14_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos', 'rewards'], "v14")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        obs_data = ds.get('observations', None)
        next_obs_data = ds.get('next_observations', None)
        terminals_data = ds.get('terminals', ds.get('dones', ds.get('dones_float', None)))

        prev_subtask_state: Optional[int] = None
        lower_entry_z_state: Optional[float] = None
        lower_entry_cube_idx_state: Optional[int] = None

        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            prev_ob_i = obs_data[i] if obs_data is not None else None
            next_ob_i = next_obs_data[i] if next_obs_data is not None else None

            prev_gp = extract_gripper_pos(prev_ob_i) if prev_ob_i is not None else None
            next_gp = extract_gripper_pos(next_ob_i) if next_ob_i is not None else None

            prev_env = self._make_env()
            prev_env.load_state(qpos_data[i], qvel=qvel, gripper_pos=prev_gp)
            next_env = self._make_env()
            next_env.load_state(next_qpos_data[i], qvel=next_qvel, gripper_pos=next_gp)

            prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev = _progress_v11(
                prev_env,
                prev_subtask=prev_subtask_state,
                prev_env=None,
                lower_entry_z=lower_entry_z_state,
                lower_entry_cube_idx=lower_entry_cube_idx_state,
            )
            next_progress, next_stage, next_subtask, lower_entry_z_next, lower_entry_cube_idx_next = _progress_v11(
                next_env,
                prev_subtask=prev_subtask_resolved,
                prev_env=prev_env,
                lower_entry_z=lower_entry_z_prev,
                lower_entry_cube_idx=lower_entry_cube_idx_prev,
            )

            shaping = shaping_coef * (discount * next_progress - prev_progress)
            stage_penalty = self._stage_penalty_from_stage(
                stage=next_stage,
                num_cubes=self.num_cubes,
                num_substages=4,
            )
            transition_event = self._v13_subtask_transition_event(
                prev_subtask=prev_subtask_resolved,
                next_subtask=next_subtask,
            )
            out[i] = stage_penalty + shaping + transition_event + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)

            prev_subtask_state = next_subtask
            lower_entry_z_state = lower_entry_z_next
            lower_entry_cube_idx_state = lower_entry_cube_idx_next
            if terminals_data is not None and bool(terminals_data[i]):
                prev_subtask_state = None
                lower_entry_z_state = None
                lower_entry_cube_idx_state = None
        return out

    def compute_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> np.ndarray:
        """Compute dense rewards for a transition dataset via version-specific handlers."""
        saved_episode_init_positions = self.episode_init_positions
        self.episode_init_positions = None
        try:
            if self.version == 'v1':
                return self.compute_v1_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
            if self.version == 'v2':
                return self.compute_v2_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
            if self.version == 'v3':
                return self.compute_v3_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
            if self.version == 'v4':
                return self.compute_v4_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus, shaping_coef=shaping_coef)
            if self.version == 'v5':
                return self.compute_v5_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus, shaping_coef=shaping_coef)
            if self.version == 'v6':
                return self.compute_v6_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus, shaping_coef=shaping_coef)
            if self.version == 'v7':
                return self.compute_v7_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v8':
                return self.compute_v8_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v9':
                return self.compute_v9_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v10':
                return self.compute_v10_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v11':
                return self.compute_v11_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v12':
                return self.compute_v12_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v13':
                return self.compute_v13_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v14':
                return self.compute_v14_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v15':
                return self.compute_v15_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v16':
                return self.compute_v16_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v17':
                return self.compute_v17_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v18':
                return self.compute_v18_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version in ('v20', 'v21'):
                return self.compute_v20_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v22':
                return self.compute_v22_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            if self.version == 'v23':
                return self.compute_v23_dataset_rewards(
                    ds,
                    discount=discount,
                    terminal_bonus=terminal_bonus,
                    shaping_coef=shaping_coef,
                )
            raise ValueError(f"Unknown version for dataset rewards: {self.version}")
        finally:
            self.episode_init_positions = saved_episode_init_positions

    def compute_v1_online_reward(
        self,
        curr_qpos: np.ndarray,
        env_reward: float = 0.0,
        curr_ob: Optional[np.ndarray] = None,
        terminal_bonus: float = 50.0,
        **_,
    ) -> float:
        return float(self.compute_potential(curr_qpos) + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v2_online_reward(
        self,
        curr_qpos: np.ndarray,
        env_reward: float = 0.0,
        curr_ob: Optional[np.ndarray] = None,
        terminal_bonus: float = 50.0,
        **_,
    ) -> float:
        return float(self.compute_potential(curr_qpos) + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v3_online_reward(
        self,
        curr_qpos: np.ndarray,
        env_reward: float = 0.0,
        curr_ob: Optional[np.ndarray] = None,
        terminal_bonus: float = 50.0,
        **_,
    ) -> float:
        gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        return float(self.compute_potential(curr_qpos, gripper_pos=gp) + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v4_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_progress, _ = self.compute_progress(prev_qpos)
        curr_progress, _ = self.compute_progress(curr_qpos)
        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        return float(env_reward + shaping + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v5_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        return float(env_reward + shaping + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v6_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        return float(env_reward + shaping + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v7_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._v7_v8_stage_penalty(qpos=curr_qpos, ob=curr_ob)
        return float(stage_penalty + shaping + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v8_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._v7_v8_stage_penalty(qpos=curr_qpos, ob=curr_ob)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v9_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        stage_penalty = self._v7_v8_stage_penalty(qpos=curr_qpos, ob=curr_ob)
        return float(stage_penalty + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v10_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._v7_v8_stage_penalty(qpos=curr_qpos, ob=curr_ob)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v11_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev = _progress_v11(
            prev_env,
            prev_subtask=self._v11_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v11_lower_entry_z,
            lower_entry_cube_idx=self._v11_lower_entry_cube_idx,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr = _progress_v11(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)

        self._v11_prev_subtask = curr_subtask
        self._v11_lower_entry_z = lower_entry_z_curr
        self._v11_lower_entry_cube_idx = lower_entry_cube_idx_curr
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v12_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v12(
            prev_env,
            prev_subtask=self._v12_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v12_lower_entry_z,
            lower_entry_cube_idx=self._v12_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v12_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v12(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)

        self._v12_prev_subtask = curr_subtask
        self._v12_lower_entry_z = lower_entry_z_curr
        self._v12_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v12_lower_entry_xy_dist = lower_entry_xy_curr
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v13_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev = _progress_v11(
            prev_env,
            prev_subtask=self._v13_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v13_lower_entry_z,
            lower_entry_cube_idx=self._v13_lower_entry_cube_idx,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr = _progress_v11(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        # transition_event = self._v13_subtask_transition_event(
        #     prev_subtask=prev_subtask_resolved,
        #     next_subtask=curr_subtask,
        # )

        self._v13_prev_subtask = curr_subtask
        self._v13_lower_entry_z = lower_entry_z_curr
        self._v13_lower_entry_cube_idx = lower_entry_cube_idx_curr
        return float(stage_penalty + shaping + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v14_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev = _progress_v11(
            prev_env,
            prev_subtask=self._v14_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v14_lower_entry_z,
            lower_entry_cube_idx=self._v14_lower_entry_cube_idx,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr = _progress_v11(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        transition_event = self._v13_subtask_transition_event(
            prev_subtask=prev_subtask_resolved,
            next_subtask=curr_subtask,
        )

        self._v14_prev_subtask = curr_subtask
        self._v14_lower_entry_z = lower_entry_z_curr
        self._v14_lower_entry_cube_idx = lower_entry_cube_idx_curr
        return float(stage_penalty + shaping + transition_event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v15_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v15(
            prev_env,
            prev_subtask=self._v15_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v15_lower_entry_z,
            lower_entry_cube_idx=self._v15_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v15_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v15(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)

        self._v15_prev_subtask = curr_subtask
        self._v15_lower_entry_z = lower_entry_z_curr
        self._v15_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v15_lower_entry_xy_dist = lower_entry_xy_curr
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v16_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v15(
            prev_env,
            prev_subtask=self._v16_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v16_lower_entry_z,
            lower_entry_cube_idx=self._v16_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v16_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v15(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)

        self._v16_prev_subtask = curr_subtask
        self._v16_lower_entry_z = lower_entry_z_curr
        self._v16_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v16_lower_entry_xy_dist = lower_entry_xy_curr
        return float(stage_penalty + shaping + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v17_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v15(
            prev_env,
            prev_subtask=self._v17_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v17_lower_entry_z,
            lower_entry_cube_idx=self._v17_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v17_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v15(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        transition_event = self._v13_subtask_transition_event(
            prev_subtask=prev_subtask_resolved,
            next_subtask=curr_subtask,
        )

        self._v17_prev_subtask = curr_subtask
        self._v17_lower_entry_z = lower_entry_z_curr
        self._v17_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v17_lower_entry_xy_dist = lower_entry_xy_curr
        return float(stage_penalty + shaping + transition_event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v18_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        prev_env = self._make_env()
        prev_env.load_state(prev_qpos, qvel=None, gripper_pos=prev_gp)
        curr_env = self._make_env()
        curr_env.load_state(curr_qpos, qvel=None, gripper_pos=curr_gp)

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v18(
            prev_env,
            prev_subtask=self._v18_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v18_lower_entry_z,
            lower_entry_cube_idx=self._v18_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v18_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v18(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)

        self._v18_prev_subtask = curr_subtask
        self._v18_lower_entry_z = lower_entry_z_curr
        self._v18_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v18_lower_entry_xy_dist = lower_entry_xy_curr
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v20_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        prev_gripper_gap_m: Optional[float] = None,
        curr_gripper_gap_m: Optional[float] = None,
        prev_left_gripper_pos: Optional[np.ndarray] = None,
        prev_right_gripper_pos: Optional[np.ndarray] = None,
        curr_left_gripper_pos: Optional[np.ndarray] = None,
        curr_right_gripper_pos: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        if prev_gripper_gap_m is not None:
            self._v20_gap_open_ref_m = max(self._v20_gap_open_ref_m, float(prev_gripper_gap_m), V20_GRIPPER_GAP_OPEN_FLOOR)
        elif curr_gripper_gap_m is not None:
            self._v20_gap_open_ref_m = max(self._v20_gap_open_ref_m, float(curr_gripper_gap_m), V20_GRIPPER_GAP_OPEN_FLOOR)

        prev_env = self._make_env()
        prev_env.load_state(
            prev_qpos,
            qvel=None,
            gripper_pos=prev_gp,
            gripper_gap_m=prev_gripper_gap_m,
            gripper_gap_open_ref_m=self._v20_gap_open_ref_m,
            left_gripper_pos=prev_left_gripper_pos,
            right_gripper_pos=prev_right_gripper_pos,
        )
        prev_env.use_pad_reach = self.version in ("v21", "v22", "v23")
        prev_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD
        curr_env = self._make_env()
        curr_env.load_state(
            curr_qpos,
            qvel=None,
            gripper_pos=curr_gp,
            gripper_gap_m=curr_gripper_gap_m,
            gripper_gap_open_ref_m=self._v20_gap_open_ref_m,
            left_gripper_pos=curr_left_gripper_pos,
            right_gripper_pos=curr_right_gripper_pos,
        )
        curr_env.use_pad_reach = self.version in ("v21", "v22", "v23")
        curr_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD

        prev_progress, _, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v18(
            prev_env,
            prev_subtask=self._v20_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v20_lower_entry_z,
            lower_entry_cube_idx=self._v20_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v20_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v18(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )

        shaping = shaping_coef * (discount * curr_progress - prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=curr_stage, num_cubes=self.num_cubes, num_substages=4)
        event = self._v8_release_event(prev_qpos=prev_qpos, curr_qpos=curr_qpos, prev_ob=prev_ob, curr_ob=curr_ob)

        self._v20_prev_subtask = curr_subtask
        self._v20_lower_entry_z = lower_entry_z_curr
        self._v20_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v20_lower_entry_xy_dist = lower_entry_xy_curr
        return float(stage_penalty + shaping + event + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v22_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        prev_gripper_gap_m: Optional[float] = None,
        curr_gripper_gap_m: Optional[float] = None,
        prev_left_gripper_pos: Optional[np.ndarray] = None,
        prev_right_gripper_pos: Optional[np.ndarray] = None,
        curr_left_gripper_pos: Optional[np.ndarray] = None,
        curr_right_gripper_pos: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        step_penalty: float = 0.0,
        **_,
    ) -> float:
        """V22/V23 online rewards: v21 semantics with monotonic anti-cycling."""
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None

        if prev_gripper_gap_m is not None:
            self._v22_gap_open_ref_m = max(self._v22_gap_open_ref_m, float(prev_gripper_gap_m), V20_GRIPPER_GAP_OPEN_FLOOR)
        elif curr_gripper_gap_m is not None:
            self._v22_gap_open_ref_m = max(self._v22_gap_open_ref_m, float(curr_gripper_gap_m), V20_GRIPPER_GAP_OPEN_FLOOR)

        prev_env = self._make_env()
        prev_env.load_state(
            prev_qpos,
            qvel=None,
            gripper_pos=prev_gp,
            gripper_gap_m=prev_gripper_gap_m,
            gripper_gap_open_ref_m=self._v22_gap_open_ref_m,
            left_gripper_pos=prev_left_gripper_pos,
            right_gripper_pos=prev_right_gripper_pos,
        )
        prev_env.use_pad_reach = True
        prev_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD
        curr_env = self._make_env()
        curr_env.load_state(
            curr_qpos,
            qvel=None,
            gripper_pos=curr_gp,
            gripper_gap_m=curr_gripper_gap_m,
            gripper_gap_open_ref_m=self._v22_gap_open_ref_m,
            left_gripper_pos=curr_left_gripper_pos,
            right_gripper_pos=curr_right_gripper_pos,
        )
        curr_env.use_pad_reach = True
        curr_env.pad_reach_threshold = V21_REACH_PAD_CUBE_DIST_THRESHOLD

        prev_progress, prev_stage, prev_subtask_resolved, lower_entry_z_prev, lower_entry_cube_idx_prev, lower_entry_xy_prev = _progress_v18(
            prev_env,
            prev_subtask=self._v22_prev_subtask,
            prev_env=None,
            lower_entry_z=self._v22_lower_entry_z,
            lower_entry_cube_idx=self._v22_lower_entry_cube_idx,
            lower_entry_xy_dist=self._v22_lower_entry_xy_dist,
        )
        curr_progress, curr_stage, curr_subtask, lower_entry_z_curr, lower_entry_cube_idx_curr, lower_entry_xy_curr = _progress_v18(
            curr_env,
            prev_subtask=prev_subtask_resolved,
            prev_env=prev_env,
            lower_entry_z=lower_entry_z_prev,
            lower_entry_cube_idx=lower_entry_cube_idx_prev,
            lower_entry_xy_dist=lower_entry_xy_prev,
        )
        mono_prev_progress, mono_curr_progress, mono_curr_stage, best_progress_next, best_stage_next = self._monotonic_progress_stage(
            prev_progress=prev_progress,
            curr_progress=curr_progress,
            prev_stage=prev_stage,
            curr_stage=curr_stage,
            best_progress=self._v22_best_progress,
            best_stage=self._v22_best_stage,
        )

        shaping = shaping_coef * (discount * mono_curr_progress - mono_prev_progress)
        stage_penalty = self._stage_penalty_from_stage(stage=mono_curr_stage, num_cubes=self.num_cubes, num_substages=4)
        self._v22_prev_subtask = curr_subtask
        self._v22_lower_entry_z = lower_entry_z_curr
        self._v22_lower_entry_cube_idx = lower_entry_cube_idx_curr
        self._v22_lower_entry_xy_dist = lower_entry_xy_curr
        self._v22_best_progress = best_progress_next
        self._v22_best_stage = best_stage_next
        return float(
            stage_penalty
            + shaping
            + self._success_bonus_post(curr_qpos, None, terminal_bonus)
            - float(step_penalty)
        )

    def compute_v23_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        prev_gripper_gap_m: Optional[float] = None,
        curr_gripper_gap_m: Optional[float] = None,
        prev_left_gripper_pos: Optional[np.ndarray] = None,
        prev_right_gripper_pos: Optional[np.ndarray] = None,
        curr_left_gripper_pos: Optional[np.ndarray] = None,
        curr_right_gripper_pos: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
        **_,
    ) -> float:
        return self.compute_v22_online_reward(
            prev_qpos=prev_qpos,
            curr_qpos=curr_qpos,
            env_reward=env_reward,
            prev_ob=prev_ob,
            curr_ob=curr_ob,
            prev_gripper_gap_m=prev_gripper_gap_m,
            curr_gripper_gap_m=curr_gripper_gap_m,
            prev_left_gripper_pos=prev_left_gripper_pos,
            prev_right_gripper_pos=prev_right_gripper_pos,
            curr_left_gripper_pos=curr_left_gripper_pos,
            curr_right_gripper_pos=curr_right_gripper_pos,
            discount=discount,
            terminal_bonus=terminal_bonus,
            shaping_coef=shaping_coef,
            step_penalty=self.v23_step_penalty,
        )

    def compute_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        prev_gripper_gap_m: Optional[float] = None,
        curr_gripper_gap_m: Optional[float] = None,
        prev_left_gripper_pos: Optional[np.ndarray] = None,
        prev_right_gripper_pos: Optional[np.ndarray] = None,
        curr_left_gripper_pos: Optional[np.ndarray] = None,
        curr_right_gripper_pos: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        shaping_coef: float = 1.0,
    ) -> float:
        """Compute dense reward for an online transition via version-specific handlers."""
        if self.version == 'v1':
            return self.compute_v1_online_reward(
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
            )
        if self.version == 'v2':
            return self.compute_v2_online_reward(
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
            )
        if self.version == 'v3':
            return self.compute_v3_online_reward(
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
            )
        if self.version == 'v4':
            return self.compute_v4_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v5':
            return self.compute_v5_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v6':
            return self.compute_v6_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v7':
            return self.compute_v7_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v8':
            return self.compute_v8_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v9':
            return self.compute_v9_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v10':
            return self.compute_v10_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v11':
            return self.compute_v11_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v12':
            return self.compute_v12_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v13':
            return self.compute_v13_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v14':
            return self.compute_v14_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v15':
            return self.compute_v15_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v16':
            return self.compute_v16_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v17':
            return self.compute_v17_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v18':
            return self.compute_v18_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version in ('v20', 'v21'):
            return self.compute_v20_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                prev_gripper_gap_m=prev_gripper_gap_m,
                curr_gripper_gap_m=curr_gripper_gap_m,
                prev_left_gripper_pos=prev_left_gripper_pos,
                prev_right_gripper_pos=prev_right_gripper_pos,
                curr_left_gripper_pos=curr_left_gripper_pos,
                curr_right_gripper_pos=curr_right_gripper_pos,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v22':
            return self.compute_v22_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                prev_gripper_gap_m=prev_gripper_gap_m,
                curr_gripper_gap_m=curr_gripper_gap_m,
                prev_left_gripper_pos=prev_left_gripper_pos,
                prev_right_gripper_pos=prev_right_gripper_pos,
                curr_left_gripper_pos=curr_left_gripper_pos,
                curr_right_gripper_pos=curr_right_gripper_pos,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        if self.version == 'v23':
            return self.compute_v23_online_reward(
                prev_qpos=prev_qpos,
                curr_qpos=curr_qpos,
                env_reward=env_reward,
                prev_ob=prev_ob,
                curr_ob=curr_ob,
                prev_gripper_gap_m=prev_gripper_gap_m,
                curr_gripper_gap_m=curr_gripper_gap_m,
                prev_left_gripper_pos=prev_left_gripper_pos,
                prev_right_gripper_pos=prev_right_gripper_pos,
                curr_left_gripper_pos=curr_left_gripper_pos,
                curr_right_gripper_pos=curr_right_gripper_pos,
                discount=discount,
                terminal_bonus=terminal_bonus,
                shaping_coef=shaping_coef,
            )
        raise ValueError(f"Unknown version for online rewards: {self.version}")
