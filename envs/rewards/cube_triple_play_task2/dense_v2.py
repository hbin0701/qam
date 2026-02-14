"""
Dense Reward V2: Distance-Based Progress

Stage determined by number of cubes at goal (same as V1).
Progress includes distance-based sub-progress for incomplete cubes.
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
        """Euclidean distance to goal."""
        return np.linalg.norm(self.position - self.goal_position)
    
    def is_at_goal(self, threshold=0.04) -> bool:
        """Check if cube is at goal position."""
        return self.distance_to_goal() < threshold
    
    def progress_to_goal(self) -> float:
        """Progress from initial to goal (0.0 to 1.0)."""
        initial_dist = np.linalg.norm(self.initial_position - self.goal_position)
        current_dist = self.distance_to_goal()
        
        if initial_dist < 1e-6:
            return 1.0
        
        progress = (initial_dist - current_dist) / initial_dist
        return np.clip(progress, 0.0, 1.0)


class Task2EnvV2:
    """Environment model for V2 reward."""
    
    def __init__(self):
        self.goal_positions = np.array([
            [0.50, 0.0, 0.02],
            [0.50, 0.2, 0.02],
            [0.50, -0.2, 0.02],
        ])
        
        self.cubes = [
            Cube(goal_position=self.goal_positions[0]),
            Cube(goal_position=self.goal_positions[1]),
            Cube(goal_position=self.goal_positions[2]),
        ]
        
        self._initialized = False
    
    def load_state(self, qpos: np.ndarray, qvel: np.ndarray = None):
        """Load state from qpos."""
        for i in range(3):
            pos_start = 7 + i * 7
            self.cubes[i].position = qpos[pos_start:pos_start+3].copy()
            if not self._initialized:
                self.cubes[i].initial_position = self.cubes[i].position.copy()
        self._initialized = True


def determination_v2(env: Task2EnvV2) -> int:
    """
    Determine stage based on number of cubes at goal.
    
    Returns:
        0-3: Number of cubes at their goal positions
    """
    num_at_goal = sum(cube.is_at_goal() for cube in env.cubes)
    return num_at_goal


def progress_v2(env: Task2EnvV2) -> Tuple[float, int]:
    """
    Compute progress for V2.
    
    Progress = completed_cubes + sum(incomplete_cube_distances)
    
    Returns:
        Tuple of (progress, stage)
    """
    stage = determination_v2(env)
    
    # Count completed cubes
    completed = 0
    incomplete_progress = 0.0
    
    for cube in env.cubes:
        if cube.is_at_goal():
            completed += 1
        else:
            # Add distance-based progress for incomplete cubes
            incomplete_progress += cube.progress_to_goal()
    
    # Total progress: completed cubes + average progress of incomplete cubes
    num_incomplete = 3 - completed
    if num_incomplete > 0:
        avg_incomplete_progress = incomplete_progress / num_incomplete
        total_progress = completed + avg_incomplete_progress
    else:
        total_progress = 3.0
    
    return (total_progress, stage)


def generate_plan_list_v2():
    """Generate plan for V2."""
    return [
        "Move cubes toward their goals",
        "Place first cube at goal",
        "Place second cube at goal",
        "Place third cube at goal",
    ]
