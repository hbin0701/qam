"""
Dense Reward V1: Simple Cube Placement Count

Simplest reward model - just counts how many cubes are at their goal positions.
Stage is determined purely by the number of cubes correctly placed.
"""

import numpy as np
from typing import Tuple


class Cube:
    """Represents a single cube."""
    def __init__(self, position=None, goal_position=None):
        self.position: np.ndarray = np.array(position) if position is not None else np.zeros(3)
        self.goal_position: np.ndarray = np.array(goal_position) if goal_position is not None else self.position.copy()
        self.size: np.ndarray = np.array([0.04, 0.04, 0.04])
    
    def distance_to_goal(self) -> float:
        """Euclidean distance to goal."""
        return np.linalg.norm(self.position - self.goal_position)
    
    def is_at_goal(self, threshold=0.04) -> bool:
        """Check if cube is at goal position."""
        return self.distance_to_goal() < threshold


class Task2EnvV1:
    """Simple environment model for V1 reward."""
    
    def __init__(self):
        self.goal_positions = np.array([
            [0.50, 0.0, 0.02],   # Cube 0
            [0.50, 0.2, 0.02],   # Cube 1
            [0.50, -0.2, 0.02],  # Cube 2
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
        self._initialized = True


def determination_v1(env: Task2EnvV1) -> int:
    """
    Determine stage based on number of cubes at goal.
    
    Returns:
        0-3: Number of cubes at their goal positions
    """
    num_at_goal = sum(cube.is_at_goal() for cube in env.cubes)
    return num_at_goal


def progress_v1(env: Task2EnvV1) -> Tuple[float, int]:
    """
    Compute progress for V1.
    
    Progress = number of cubes at goal (0.0 to 3.0)
    
    Returns:
        Tuple of (progress, stage)
    """
    stage = determination_v1(env)
    progress = float(stage)
    return (progress, stage)


def generate_plan_list_v1():
    """Generate plan for V1."""
    return [
        "Place first cube at goal",
        "Place second cube at goal",
        "Place third cube at goal",
    ]
