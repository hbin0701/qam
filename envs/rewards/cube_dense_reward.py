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
# - Typical gripper-cube median distance is around 0.14-0.19.
# - Typical "closed" gripper width is around 0.027 (so 0.02 is too strict).
V6_REACH_THRESHOLD = 0.10
V6_GOAL_APPROACH_THRESHOLD = 0.09
V6_GRIPPER_OPEN_WIDTH = 0.08
V6_GRASP_WIDTH_THRESHOLD = 0.045
V6_GRASP_LIFT_THRESHOLD = 0.008
V6_GRASP_LIFT_TARGET = 0.02


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
    def __init__(self, position=None, goal_position=None):
        self.position: np.ndarray = np.array(position) if position is not None else np.zeros(3)
        self.initial_position: np.ndarray = self.position.copy()
        self.goal_position: np.ndarray = np.array(goal_position) if goal_position is not None else self.position.copy()

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - self.goal_position))

    def is_at_goal(self, threshold=CUBE_SUCCESS_THRESHOLD) -> bool:
        return self.distance_to_goal() <= threshold

    def progress_to_goal(self, threshold=CUBE_SUCCESS_THRESHOLD) -> float:
        """Progress from initial position to goal (0.0 to 1.0), saturating at success threshold."""
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
                 cube_order: Optional[List[int]] = None):
        self.num_cubes = num_cubes
        self.goal_positions = goal_positions
        self.cubes: List[CubeState] = [
            CubeState(goal_position=goal_positions[i]) for i in range(num_cubes)
        ]
        # Set task initial positions for V2 distance-based progress
        if init_positions is not None:
            for i in range(num_cubes):
                self.cubes[i].initial_position = init_positions[i].copy()
        # Fixed cube ordering for v3/v5 sequential progress (nearest-first from gripper home)
        self.cube_order: List[int] = cube_order if cube_order is not None else list(range(num_cubes))
        self.gripper_width: float = 0.0
        self.gripper_pos: Optional[np.ndarray] = None

    def load_state(self, qpos: np.ndarray, qvel: np.ndarray = None,
                   gripper_pos: Optional[np.ndarray] = None):
        """Load state from qpos vector.

        qpos layout: [arm_joints(6), gripper_joints(8), cube0_pos(3)+quat(4), cube1_pos(3)+quat(4), ...]
        gripper_pos: end-effector position (from observation), needed for v3/v5 reach substage.
        """
        for i in range(self.num_cubes):
            pos_start = 14 + i * 7
            self.cubes[i].position = qpos[pos_start:pos_start + 3].copy()
        self.gripper_width = qpos[6] * 2
        if gripper_pos is not None:
            self.gripper_pos = gripper_pos.copy()

    def is_grasped(self, cube_idx: int) -> bool:
        cube = self.cubes[cube_idx]
        gripper_closed = self.gripper_width < 0.02
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
        effective_current = max(current_dist - CUBE_SUCCESS_THRESHOLD, 0.0)
        effective_init = max(place_init_dist - CUBE_SUCCESS_THRESHOLD, 1e-6)
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
    cube_goal_dist = cube.distance_to_goal()
    cube_lift = float(cube.position[2] - cube.initial_position[2])
    gripper_open = float(np.clip(env.gripper_width / V6_GRIPPER_OPEN_WIDTH, 0.0, 1.0))
    # Use v6-specific grasp condition (less strict than env.is_grasped()).
    grasped = (env.gripper_width < V6_GRASP_WIDTH_THRESHOLD) and (cube_lift > V6_GRASP_LIFT_THRESHOLD)

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
        close_progress = float(np.clip((V6_GRIPPER_OPEN_WIDTH - env.gripper_width) / V6_GRIPPER_OPEN_WIDTH, 0.0, 1.0))
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
    ):
        self.task_name = task_name
        self.version = version
        self.debug = debug

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

        # Precompute cube ordering for v3/v5/v6: nearest cube to gripper home goes first
        if self.init_positions is not None:
            self.cube_order = compute_cube_order(self.init_positions)
        else:
            self.cube_order = list(range(self.num_cubes))

        if version not in ('v1', 'v2', 'v3', 'v4', 'v5', 'v6'):
            raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'")

        if debug:
            print(f"[DenseReward-{version}] env_type={self.env_type}, task_id={self.task_id}, "
                  f"num_cubes={self.num_cubes}")
            print(f"[DenseReward-{version}] goal_positions={self.goal_positions.tolist()}")
            print(f"[DenseReward-{version}] cube_order={self.cube_order}")

    @property
    def max_progress(self) -> float:
        """Maximum raw progress value for this version."""
        if self.version in ('v3', 'v5', 'v6'):
            return float(self.num_cubes)
        return float(self.num_cubes)

    def compute_potential(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None,
                          gripper_pos: Optional[np.ndarray] = None) -> float:
        """Compute shaping potential/value from progress.

        - v2/v3 use raw progress P_v2(s)/P_v3(s), range [0, num_cubes].
        - other versions keep centered form, range [-num_cubes, 0].
        """
        progress, _ = self.compute_progress(qpos, qvel, gripper_pos)
        if self.version in ('v2', 'v3', 'v6'):
            return float(progress)
        scale = self.num_cubes / self.max_progress
        return float(progress * scale - self.num_cubes)

    def _make_env(self) -> CubeEnvModel:
        return CubeEnvModel(self.num_cubes, self.goal_positions, self.init_positions, self.cube_order)

    def compute_progress(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None,
                         gripper_pos: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """Compute progress for a single state."""
        env = self._make_env()
        env.load_state(qpos, qvel, gripper_pos)
        return PROGRESS_FNS[self.version](env)

    def _require_keys(self, ds: Dict[str, np.ndarray], keys: List[str], ctx: str):
        missing = [k for k in keys if k not in ds]
        if missing:
            raise ValueError(f"{ctx} requires keys: {missing}")

    def _is_success_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> bool:
        """Task success from post-state only (all cubes at goal)."""
        env = self._make_env()
        env.load_state(qpos, qvel=qvel, gripper_pos=None)
        return all(c.is_at_goal() for c in env.cubes)

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

    def compute_v4_dataset_rewards(self, ds: Dict[str, np.ndarray], discount: float = 0.99, terminal_bonus: float = 1.0) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos'], "v4")
        qpos_data = ds['qpos']
        qvel_data = ds.get('qvel', None)
        next_qpos_data = ds['next_qpos']
        next_qvel_data = ds.get('next_qvel', None)
        out = np.zeros(len(qpos_data), dtype=np.float32)
        for i in range(len(qpos_data)):
            qvel = qvel_data[i] if qvel_data is not None else None
            next_qvel = next_qvel_data[i] if next_qvel_data is not None else None
            curr_progress, _ = self.compute_progress(qpos_data[i], qvel)
            next_progress, _ = self.compute_progress(next_qpos_data[i], next_qvel)
            out[i] = (discount * next_progress - curr_progress) + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v5_dataset_rewards(self, ds: Dict[str, np.ndarray], discount: float = 0.99, terminal_bonus: float = 1.0) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos'], "v5")
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
            out[i] = (discount * next_progress - curr_progress) + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_v6_dataset_rewards(self, ds: Dict[str, np.ndarray], discount: float = 0.99, terminal_bonus: float = 1.0) -> np.ndarray:
        self._require_keys(ds, ['qpos', 'next_qpos'], "v6")
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
            out[i] = (discount * next_progress - curr_progress) + self._success_bonus_post(next_qpos_data[i], next_qvel, terminal_bonus)
        return out

    def compute_dataset_rewards(
        self,
        ds: Dict[str, np.ndarray],
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
    ) -> np.ndarray:
        """Compute dense rewards for a transition dataset via version-specific handlers."""
        if self.version == 'v1':
            return self.compute_v1_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
        if self.version == 'v2':
            return self.compute_v2_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
        if self.version == 'v3':
            return self.compute_v3_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
        if self.version == 'v4':
            return self.compute_v4_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
        if self.version == 'v5':
            return self.compute_v5_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
        if self.version == 'v6':
            return self.compute_v6_dataset_rewards(ds, discount=discount, terminal_bonus=terminal_bonus)
        raise ValueError(f"Unknown version for dataset rewards: {self.version}")

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
        **_,
    ) -> float:
        prev_progress, _ = self.compute_progress(prev_qpos)
        curr_progress, _ = self.compute_progress(curr_qpos)
        return float((discount * curr_progress - prev_progress) + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v5_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        return float((discount * curr_progress - prev_progress) + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_v6_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
        **_,
    ) -> float:
        prev_gp = extract_gripper_pos(prev_ob) if prev_ob is not None else None
        curr_gp = extract_gripper_pos(curr_ob) if curr_ob is not None else None
        prev_progress, _ = self.compute_progress(prev_qpos, gripper_pos=prev_gp)
        curr_progress, _ = self.compute_progress(curr_qpos, gripper_pos=curr_gp)
        return float((discount * curr_progress - prev_progress) + self._success_bonus_post(curr_qpos, None, terminal_bonus))

    def compute_online_reward(
        self,
        prev_qpos: np.ndarray,
        curr_qpos: np.ndarray,
        env_reward: float,
        prev_ob: Optional[np.ndarray] = None,
        curr_ob: Optional[np.ndarray] = None,
        discount: float = 0.99,
        terminal_bonus: float = 1.0,
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
            )
        raise ValueError(f"Unknown version for online rewards: {self.version}")
