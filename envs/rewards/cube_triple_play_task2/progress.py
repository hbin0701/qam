"""
Dense Reward Model for OGBench Cube-Triple Task 2 (Triple Pick and Place)

Task 2 Objective:
- Move 3 cubes from initial positions to goal positions
- Initial: [0.35, -0.2, 0.02], [0.35, 0.0, 0.02], [0.35, 0.2, 0.02]
- Goal:    [0.50, 0.0, 0.02], [0.50, 0.2, 0.02], [0.50, -0.2, 0.02]

This implements physics-based progress tracking similar to dsrl_pi0's generated_progress.py
"""

import numpy as np
from typing import Dict, Tuple


class Cube:
    """Represents a single cube in the environment."""
    def __init__(self, position=None, size=None, goal_position=None):
        self.position: np.ndarray = np.array(position) if position is not None else np.zeros(3)
        self.initial_position: np.ndarray = self.position.copy()
        self.goal_position: np.ndarray = np.array(goal_position) if goal_position is not None else self.position.copy()
        # Standard cube size in OGBench (4cm x 4cm x 4cm)
        self.size: np.ndarray = np.array(size) if size is not None else np.array([0.04, 0.04, 0.04])
    
    def get_bottom_z(self) -> float:
        """Returns the z-coordinate of the bottom surface."""
        return self.position[2] - self.size[2] / 2
    
    def get_top_z(self) -> float:
        """Returns the z-coordinate of the top surface."""
        return self.position[2] + self.size[2] / 2
    
    def distance_to_goal(self) -> float:
        """Euclidean distance from current position to goal."""
        return np.linalg.norm(self.position - self.goal_position)
    
    def is_at_goal(self, threshold=0.04) -> bool:
        """Check if cube is at its goal position."""
        return self.distance_to_goal() < threshold


class Robot:
    """Represents the UR5e robot arm."""
    def __init__(self, position=None):
        self.ee_position: np.ndarray = np.array(position) if position is not None else np.zeros(3)
        self.gripper_width: float = 0.0  # 0 = closed, ~0.08 = fully open
    
    @property
    def tcp(self) -> np.ndarray:
        """Tool center point (gripper center)."""
        return self.ee_position


class Task2Env:
    """
    Environment model for Task 2: Triple Pick and Place.
    
    Tracks the state of 3 cubes and the robot, computing progress through the task.
    """
    
    def __init__(self):
        # Define goal positions for Task 2
        self.goal_positions = np.array([
            [0.50, 0.0, 0.02],   # Cube 0 goal
            [0.50, 0.2, 0.02],   # Cube 1 goal
            [0.50, -0.2, 0.02],  # Cube 2 goal
        ])
        
        # Initialize objects
        self.robot = Robot()
        self.cubes = [
            Cube(goal_position=self.goal_positions[0]),
            Cube(goal_position=self.goal_positions[1]),
            Cube(goal_position=self.goal_positions[2]),
        ]
        
        # State tracking
        self.max_progress_seen = 0.0
        self.current_stage = 0
        self.subtask_initial_states = {}
        self._initialized = False
        
        # Track which cube we're currently working on
        self.active_cube_idx = 0
        self.cubes_completed = [False, False, False]
    
    def load_state(self, qpos: np.ndarray, qvel: np.ndarray = None):
        """
        Load state from OGBench qpos array.
        
        OGBench qpos structure for cube-triple:
        - [0:6]: Robot arm joint positions
        - [6]: Gripper position
        - [7:10]: Cube 0 position (x, y, z)
        - [10:14]: Cube 0 orientation (quaternion)
        - [14:17]: Cube 1 position
        - [17:21]: Cube 1 orientation
        - [21:24]: Cube 2 position
        - [24:28]: Cube 2 orientation
        """
        # Extract cube positions
        for i in range(3):
            pos_start = 7 + i * 7  # Each cube has 7 values (3 pos + 4 quat)
            self.cubes[i].position = qpos[pos_start:pos_start+3].copy()
            if not self._initialized:
                self.cubes[i].initial_position = self.cubes[i].position.copy()
        
        # Extract gripper state (simplified - just the width)
        self.robot.gripper_width = qpos[6] * 2  # Approximate gripper opening
        
        # For robot EE position, we'd need forward kinematics
        # For now, we'll approximate it based on the active cube
        # In practice, you might compute this from joint angles
        if self.active_cube_idx < 3:
            # Approximate: robot is near the cube it's working on
            active_cube = self.cubes[self.active_cube_idx]
            self.robot.ee_position = active_cube.position + np.array([0, 0, 0.1])
        
        self._initialized = True
    
    def is_grasped(self, cube_idx: int) -> bool:
        """
        Check if a cube is grasped by the robot.
        
        Heuristic: gripper is closed and cube is elevated.
        """
        cube = self.cubes[cube_idx]
        gripper_closed = self.robot.gripper_width < 0.02
        is_lifted = cube.position[2] > cube.initial_position[2] + 0.01
        return gripper_closed and is_lifted
    
    def is_cube_moving_to_goal(self, cube_idx: int) -> bool:
        """Check if cube is being transported toward its goal."""
        cube = self.cubes[cube_idx]
        # Cube is lifted and closer to goal than initial position
        is_lifted = cube.position[2] > 0.04  # Above table
        initial_dist = np.linalg.norm(cube.initial_position - cube.goal_position)
        current_dist = cube.distance_to_goal()
        return is_lifted and current_dist < initial_dist * 0.8


def determination_task2(env: Task2Env) -> int:
    """
    Determine the current stage of Task 2 execution.
    
    Stages (per cube, cycling through 3 cubes):
    0: Idle / Approaching cube
    1: Grasping cube
    2: Lifting and moving cube
    3: Placing cube at goal
    4: Releasing cube
    5: All cubes placed (terminal)
    
    Returns:
        Stage index (0-5)
    """
    # Check if all cubes are at their goals
    all_complete = all(cube.is_at_goal() for cube in env.cubes)
    if all_complete:
        return 5
    
    # Find the active cube (first one not at goal)
    for i, cube in enumerate(env.cubes):
        if not cube.is_at_goal():
            env.active_cube_idx = i
            break
    
    active_cube = env.cubes[env.active_cube_idx]
    
    # Stage determination for the active cube
    if active_cube.is_at_goal():
        # This cube is done, move to next
        env.cubes_completed[env.active_cube_idx] = True
        return 0  # Reset to approach next cube
    
    # Check if cube is grasped
    if env.is_grasped(env.active_cube_idx):
        # Cube is grasped
        if env.is_cube_moving_to_goal(env.active_cube_idx):
            # Moving toward goal
            if active_cube.distance_to_goal() < 0.08:
                # Close to goal, in placing phase
                if active_cube.position[2] < 0.04:
                    # Lowered to table
                    return 4  # Releasing
                return 3  # Placing
            return 2  # Lifting/moving
        return 1  # Just grasped
    
    # Not grasped yet
    robot_to_cube_dist = np.linalg.norm(env.robot.tcp[:2] - active_cube.position[:2])
    if robot_to_cube_dist < 0.05:
        return 1  # Approaching/grasping
    
    return 0  # Idle/approaching


def progress_task2(env: Task2Env) -> Tuple[float, int]:
    """
    Compute progress for Task 2.
    
    Progress is computed as:
    - Each cube contributes up to 1.0 progress when placed
    - Within each cube, progress is divided into sub-stages
    - Total progress range: [0.0, 3.0] (3 cubes)
    
    Returns:
        Tuple of (progress, stage_index)
    """
    stage = determination_task2(env)
    
    # Count completed cubes
    num_completed = sum(env.cubes_completed)
    base_progress = float(num_completed)
    
    # Compute sub-progress for the active cube
    if env.active_cube_idx < 3:
        active_cube = env.cubes[env.active_cube_idx]
        subprogress = 0.0
        
        if stage == 0:
            # Approaching: progress based on robot distance to cube
            if 0 not in env.subtask_initial_states:
                env.subtask_initial_states[0] = {
                    'initial_dist': np.linalg.norm(env.robot.tcp - active_cube.position)
                }
            initial_dist = env.subtask_initial_states[0]['initial_dist']
            current_dist = np.linalg.norm(env.robot.tcp - active_cube.position)
            subprogress = np.clip((initial_dist - current_dist) / (initial_dist + 1e-6), 0.0, 1.0)
        
        elif stage == 1:
            # Grasping: progress based on gripper closing
            if 1 not in env.subtask_initial_states:
                env.subtask_initial_states[1] = {
                    'initial_width': env.robot.gripper_width
                }
            initial_width = env.subtask_initial_states[1]['initial_width']
            close_progress = np.clip((initial_width - env.robot.gripper_width) / (initial_width + 1e-6), 0.0, 1.0)
            subprogress = close_progress
        
        elif stage == 2:
            # Lifting and moving: combined lift + horizontal movement
            if 2 not in env.subtask_initial_states:
                env.subtask_initial_states[2] = {
                    'initial_z': active_cube.initial_position[2],
                    'initial_dist_to_goal': active_cube.distance_to_goal()
                }
            initial_z = env.subtask_initial_states[2]['initial_z']
            initial_dist = env.subtask_initial_states[2]['initial_dist_to_goal']
            
            # Lift progress
            lift_progress = np.clip((active_cube.position[2] - initial_z) / 0.05, 0.0, 1.0)
            
            # Horizontal movement progress
            current_dist = active_cube.distance_to_goal()
            move_progress = np.clip((initial_dist - current_dist) / (initial_dist + 1e-6), 0.0, 1.0)
            
            subprogress = 0.3 * lift_progress + 0.7 * move_progress
        
        elif stage == 3:
            # Placing: lowering to goal
            if 3 not in env.subtask_initial_states:
                env.subtask_initial_states[3] = {
                    'initial_z': active_cube.position[2]
                }
            initial_z = env.subtask_initial_states[3]['initial_z']
            target_z = 0.02  # Table height
            lower_progress = np.clip((initial_z - active_cube.position[2]) / (initial_z - target_z + 1e-6), 0.0, 1.0)
            subprogress = lower_progress
        
        elif stage == 4:
            # Releasing: gripper opening
            open_progress = np.clip(env.robot.gripper_width / 0.08, 0.0, 1.0)
            subprogress = open_progress
        
        elif stage == 5:
            subprogress = 1.0
        
        # Each stage contributes 0.2 to the cube's progress (5 stages = 1.0 per cube)
        cube_progress = (stage / 5.0) + (subprogress / 5.0)
        total_progress = base_progress + np.clip(cube_progress, 0.0, 1.0)
    else:
        total_progress = 3.0  # All cubes done
    
    env.max_progress_seen = max(env.max_progress_seen, total_progress)
    env.current_stage = stage
    
    return (env.max_progress_seen, stage)


def generate_plan_list_task2():
    """Generate human-readable plan for Task 2."""
    return [
        "CUBE 1: Approach and grasp first cube",
        "CUBE 1: Lift and move to goal position",
        "CUBE 1: Place and release at goal",
        "CUBE 2: Approach and grasp second cube",
        "CUBE 2: Lift and move to goal position",
        "CUBE 2: Place and release at goal",
        "CUBE 3: Approach and grasp third cube",
        "CUBE 3: Lift and move to goal position",
        "CUBE 3: Place and release at goal",
    ]
