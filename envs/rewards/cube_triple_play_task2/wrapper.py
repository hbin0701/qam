"""
Unified Dense Reward Wrapper for QAM

Supports multiple reward model versions:
- v1: Simple cube placement count
- v2: Distance-based progress
- v3: Detailed stage tracking with strict ordering
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

# Add envs directory to path
_envs_dir = Path(__file__).parent
if str(_envs_dir) not in sys.path:
    sys.path.insert(0, str(_envs_dir))

# Import all versions
from dense_v1 import Task2EnvV1, progress_v1, generate_plan_list_v1
from dense_v2 import Task2EnvV2, progress_v2, generate_plan_list_v2
from dense_v3 import Task2EnvV3, progress_v3, generate_plan_list_v3


class DenseRewardWrapper:
    """
    Unified wrapper for dense reward computation in QAM.
    
    Usage:
        # Use V1 (simple)
        wrapper = DenseRewardWrapper(version="v1")
        
        # Use V2 (distance-based)
        wrapper = DenseRewardWrapper(version="v2")
        
        # Use V3 (detailed)
        wrapper = DenseRewardWrapper(version="v3")

        reward = wrapper.compute_reward(qpos_prev, qpos_curr)
    """
    
    def __init__(
        self,
        task_name: str = "cube-triple-play-singletask-task2-v0",
        version: str = "v2",
        debug: bool = False,
    ):
        """
        Initialize dense reward wrapper.

        Args:
            task_name: OGBench task name
            version: Reward model version ("v1", "v2", or "v3")
            debug: Whether to print debug information
        """
        self.task_name = task_name
        self.version = version
        self.debug = debug

        # Select version-specific components
        if version == "v1":
            self.env_class = Task2EnvV1
            self.progress_fn = progress_v1
            self.plan_list = generate_plan_list_v1()
        elif version == "v2":
            self.env_class = Task2EnvV2
            self.progress_fn = progress_v2
            self.plan_list = generate_plan_list_v2()
        elif version == "v3":
            self.env_class = Task2EnvV3
            self.progress_fn = progress_v3
            self.plan_list = generate_plan_list_v3()
        else:
            raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3'")

        if debug:
            print(f"[DenseReward-{version}] Initialized for task: {task_name}")
            print(f"[DenseReward-{version}] Plan: {self.plan_list}")

        self.reset()
    
    def reset(self):
        """Reset internal state for a new episode."""
        self._env = self.env_class()
        self._last_progress = 0.0
        if self.debug:
            print(f"[DenseReward-{self.version}] Wrapper state reset.")
    
    def compute_progress(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """
        Compute progress and stage for a single state.
        
        Args:
            qpos: Joint positions and object states
            qvel: Joint velocities (optional)
            
        Returns:
            Tuple of (progress, stage_index)
        """
        # Create a fresh environment for stateless computation
        temp_env = self.env_class()
        temp_env.load_state(qpos, qvel)
        progress, stage_idx = self.progress_fn(temp_env)
        return float(progress), int(stage_idx)
    
    def compute_reward(
        self,
        qpos_prev: np.ndarray,
        qpos_curr: np.ndarray,
        qvel_prev: Optional[np.ndarray] = None,
        qvel_curr: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """Compute dense reward as progress delta."""
        progress_prev, _ = self.compute_progress(qpos_prev, qvel_prev)
        progress_curr, stage_idx = self.compute_progress(qpos_curr, qvel_curr)

        reward = float(progress_curr - progress_prev)

        self._last_progress = progress_curr

        info = {
            f"dense_{self.version}_stage": stage_idx,
            f"dense_{self.version}_progress": progress_curr,
            f"dense_{self.version}_reward": reward,
        }

        if self.debug:
            print(f"[DenseReward-{self.version}] Progress: {progress_prev:.4f} -> {progress_curr:.4f}, "
                  f"Reward: {reward:.4f}, Stage: {stage_idx}")

        return reward, info

    def compute_reward_batch(
        self,
        qpos_prev_batch: np.ndarray,
        qpos_curr_batch: np.ndarray,
        qvel_prev_batch: Optional[np.ndarray] = None,
        qvel_curr_batch: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, list]:
        """Compute rewards for a batch of transitions."""
        batch_size = qpos_prev_batch.shape[0]
        rewards = []
        infos = []

        for i in range(batch_size):
            qvel_prev = qvel_prev_batch[i] if qvel_prev_batch is not None else None
            qvel_curr = qvel_curr_batch[i] if qvel_curr_batch is not None else None
            reward, info = self.compute_reward(
                qpos_prev_batch[i], qpos_curr_batch[i], qvel_prev, qvel_curr
            )
            rewards.append(reward)
            infos.append(info)

        return np.array(rewards), infos


def test_all_versions():
    """Test all three reward versions with synthetic data."""
    print("=" * 60)
    print("Testing All Dense Reward Versions")
    print("=" * 60)
    
    # Create synthetic qpos data
    qpos_dim = 7 + 3 * 7  # = 28
    
    # Initial state: cubes at starting positions
    qpos_prev = np.zeros(qpos_dim)
    qpos_prev[6] = 0.04  # Gripper open
    qpos_prev[7:10] = [0.35, -0.2, 0.02]  # Cube 0 initial
    qpos_prev[14:17] = [0.35, 0.0, 0.02]  # Cube 1 initial
    qpos_prev[21:24] = [0.35, 0.2, 0.02]  # Cube 2 initial
    
    # Current state: first cube moved toward goal
    qpos_curr = qpos_prev.copy()
    qpos_curr[7:10] = [0.45, -0.1, 0.05]  # Cube 0 moving toward goal
    qpos_curr[6] = 0.01  # Gripper closed
    
    # Test each version
    for version in ["v1", "v2", "v3"]:
        print(f"\n{'='*60}")
        print(f"Testing Version: {version}")
        print(f"{'='*60}")

        wrapper = DenseRewardWrapper(
            task_name="cube-triple-play-singletask-task2-v0",
            version=version,
            debug=True
        )

        reward, info = wrapper.compute_reward(qpos_prev, qpos_curr)
        print(f"\n✓ {version.upper()} Reward: {reward:.4f}")
        print(f"  Info: {info}")
    
    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_all_versions()
