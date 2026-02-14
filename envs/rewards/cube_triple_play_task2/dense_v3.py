"""
Dense Reward V3: Detailed Stage Tracking with Strict Ordering

Most detailed reward model with strict cube ordering enforcement.
Tracks: Reach & Grasp -> Carry -> Place for each cube sequentially.
"""

import numpy as np
from typing import Tuple


class Cube:
    """Represents a single cube."""
    def __init__(self, position=None, goal_position=None):
        self.position: np.ndarray = np.array(position) if position is not None else np.zeros(3)
        self.initial_position: np.ndarray = self.position.copy()
        self.goal_position: np.ndarray = np.array(goal_position) if goal_position is not None else self.position.copy()
        self.size: np.ndarray = np.array([0.04, 0.04, 0.04])
    
    def distance_to_goal(self) -> float:
        return np.linalg.norm(self.position - self.goal_position)
    
    def is_at_goal(self, threshold=0.04) -> bool:
        return self.distance_to_goal() < threshold
    
    def get_bottom_z(self) -> float:
        return self.position[2] - self.size[2] / 2


class Robot:
    """Robot arm model."""
    def __init__(self):
        self.ee_position: np.ndarray = np.zeros(3)
        self.gripper_width: float = 0.0
    
    @property
    def tcp(self) -> np.ndarray:
        return self.ee_position


class Task2EnvV3:
    """Detailed environment model for V3 reward."""
    
    def __init__(self):
        self.goal_positions = np.array([
            [0.50, 0.0, 0.02],
            [0.50, 0.2, 0.02],
            [0.50, -0.2, 0.02],
        ])
        
        self.robot = Robot()
        self.cubes = [
            Cube(goal_position=self.goal_positions[0]),
            Cube(goal_position=self.goal_positions[1]),
            Cube(goal_position=self.goal_positions[2]),
        ]
        
        # Strict ordering: must complete cube 0, then 1, then 2
        self.active_cube_idx = 0
        self.cubes_completed = [False, False, False]
        
        # Sub-stage tracking for current cube
        self.current_substage = 0  # 0: reach_grasp, 1: carry, 2: place
        
        # State for sub-progress computation
        self.subtask_initial_states = {}
        self._initialized = False
    
    def load_state(self, qpos: np.ndarray, qvel: np.ndarray = None):
        """Load state from qpos."""
        for i in range(3):
            pos_start = 7 + i * 7
            self.cubes[i].position = qpos[pos_start:pos_start+3].copy()
            if not self._initialized:
                self.cubes[i].initial_position = self.cubes[i].position.copy()
        
        # Gripper state
        self.robot.gripper_width = qpos[6] * 2
        
        # Approximate EE position (near active cube)
        if self.active_cube_idx < 3:
            active_cube = self.cubes[self.active_cube_idx]
            self.robot.ee_position = active_cube.position + np.array([0, 0, 0.1])
        
        self._initialized = True
    
    def is_grasped(self, cube_idx: int) -> bool:
        """Check if cube is grasped."""
        cube = self.cubes[cube_idx]
        gripper_closed = self.robot.gripper_width < 0.02
        is_lifted = cube.position[2] > cube.initial_position[2] + 0.01
        return gripper_closed and is_lifted


def determination_v3(env: Task2EnvV3) -> int:
    """
    Determine detailed stage with strict cube ordering.
    
    Stage encoding: cube_idx * 3 + substage
    - Substages: 0=reach_grasp, 1=carry, 2=place
    
    Returns:
        0-8: Stage index (3 cubes × 3 substages)
        9: All complete
    """
    # Find active cube (first incomplete)
    for i in range(3):
        if not env.cubes[i].is_at_goal():
            env.active_cube_idx = i
            break
    else:
        # All cubes complete
        return 9
    
    active_cube = env.cubes[env.active_cube_idx]
    
    # Determine substage for active cube
    if active_cube.is_at_goal():
        # This cube is done
        substage = 2
    elif env.is_grasped(env.active_cube_idx):
        # Cube is grasped
        if active_cube.distance_to_goal() < 0.08:
            # Near goal, placing
            substage = 2
        else:
            # Carrying toward goal
            substage = 1
    else:
        # Not grasped yet - reaching and grasping
        substage = 0
    
    stage = env.active_cube_idx * 3 + substage
    env.current_substage = substage
    return stage


def progress_v3(env: Task2EnvV3) -> Tuple[float, int]:
    """
    Compute detailed progress for V3.
    
    Progress range: 0.0 to 9.0 (3 cubes × 3 substages)
    Each substage contributes 0.0 to 1.0 progress.
    
    Returns:
        Tuple of (progress, stage)
    """
    stage = determination_v3(env)
    
    if stage >= 9:
        return (9.0, stage)
    
    # Base progress from completed cubes
    base_progress = float(env.active_cube_idx * 3)
    
    # Sub-progress within current substage
    active_cube = env.cubes[env.active_cube_idx]
    substage = env.current_substage
    subprogress = 0.0
    
    if substage == 0:
        # Reach & Grasp: combined progress
        key = f"{env.active_cube_idx}_reach_grasp"
        if key not in env.subtask_initial_states:
            env.subtask_initial_states[key] = {
                'initial_dist': np.linalg.norm(env.robot.tcp - active_cube.position),
                'initial_width': env.robot.gripper_width
            }
        
        # Progress toward cube
        initial_dist = env.subtask_initial_states[key]['initial_dist']
        current_dist = np.linalg.norm(env.robot.tcp - active_cube.position)
        reach_progress = np.clip((initial_dist - current_dist) / (initial_dist + 1e-6), 0.0, 1.0)
        
        # Gripper closing progress
        initial_width = env.subtask_initial_states[key]['initial_width']
        grasp_progress = np.clip((initial_width - env.robot.gripper_width) / (initial_width + 1e-6), 0.0, 1.0)
        
        # Combined: 60% reach, 40% grasp
        subprogress = 0.6 * reach_progress + 0.4 * grasp_progress
    
    elif substage == 1:
        # Carry: progress based on distance to goal
        key = f"{env.active_cube_idx}_carry"
        if key not in env.subtask_initial_states:
            env.subtask_initial_states[key] = active_cube.distance_to_goal()
        initial_dist = env.subtask_initial_states[key]
        current_dist = active_cube.distance_to_goal()
        subprogress = np.clip((initial_dist - current_dist) / (initial_dist + 1e-6), 0.0, 1.0)
    
    elif substage == 2:
        # Place: progress based on lowering to goal height
        key = f"{env.active_cube_idx}_place"
        if key not in env.subtask_initial_states:
            env.subtask_initial_states[key] = active_cube.position[2]
        initial_z = env.subtask_initial_states[key]
        target_z = 0.02
        current_z = active_cube.position[2]
        subprogress = np.clip((initial_z - current_z) / (initial_z - target_z + 1e-6), 0.0, 1.0)
    
    total_progress = base_progress + substage + np.clip(subprogress, 0.0, 1.0)
    return (total_progress, stage)


def generate_plan_list_v3():
    """Generate detailed plan for V3."""
    return [
        "CUBE 0: Reach & Grasp",
        "CUBE 0: Carry to goal",
        "CUBE 0: Place at goal",
        "CUBE 1: Reach & Grasp",
        "CUBE 1: Carry to goal",
        "CUBE 1: Place at goal",
        "CUBE 2: Reach & Grasp",
        "CUBE 2: Carry to goal",
        "CUBE 2: Place at goal",
    ]
