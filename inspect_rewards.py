#!/usr/bin/env python3
"""
Quick script to inspect OGBench dataset reward format.
Usage: python inspect_rewards.py [dataset_path]
"""

import numpy as np
import sys
import os

def inspect_rewards(dataset_path):
    """Inspect reward structure in an OGBench dataset."""
    
    # Expand path
    dataset_path = os.path.expanduser(dataset_path)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    print(f"Loading dataset: {dataset_path}")
    print("=" * 60)
    
    # Load dataset
    data = np.load(dataset_path)
    
    print("\nDataset keys:", list(data.keys()))
    
    # Inspect rewards
    rewards = data['rewards']
    terminals = data.get('terminals', None)
    
    print(f"\nReward array shape: {rewards.shape}")
    print(f"Reward dtype: {rewards.dtype}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("REWARD STATISTICS")
    print("=" * 60)
    print(f"Total transitions: {len(rewards)}")
    print(f"Min reward: {rewards.min():.6f}")
    print(f"Max reward: {rewards.max():.6f}")
    print(f"Mean reward: {rewards.mean():.6f}")
    print(f"Std reward: {rewards.std():.6f}")
    
    # Unique values
    unique_rewards = np.unique(rewards)
    print(f"\nNumber of unique reward values: {len(unique_rewards)}")
    print(f"Unique values: {unique_rewards[:20]}")  # Show first 20
    
    # Zero vs non-zero
    num_zero = np.sum(rewards == 0)
    num_nonzero = np.sum(rewards != 0)
    print(f"\nZero rewards: {num_zero} ({100*num_zero/len(rewards):.2f}%)")
    print(f"Non-zero rewards: {num_nonzero} ({100*num_nonzero/len(rewards):.2f}%)")
    
    # Sample non-zero rewards
    if num_nonzero > 0:
        nonzero_rewards = rewards[rewards != 0]
        print(f"\nSample non-zero rewards (first 20):")
        print(nonzero_rewards[:20])
        print(f"Non-zero reward stats: min={nonzero_rewards.min():.6f}, "
              f"max={nonzero_rewards.max():.6f}, mean={nonzero_rewards.mean():.6f}")
    
    # Check if rewards are only at episode ends
    if terminals is not None:
        print("\n" + "=" * 60)
        print("EPISODE STRUCTURE")
        print("=" * 60)
        terminal_indices = np.where(terminals > 0)[0]
        print(f"Number of episodes: {len(terminal_indices)}")
        
        # Check rewards at terminal states
        if len(terminal_indices) > 0:
            rewards_at_terminals = rewards[terminal_indices]
            print(f"\nRewards at terminal states:")
            print(f"  Min: {rewards_at_terminals.min():.6f}")
            print(f"  Max: {rewards_at_terminals.max():.6f}")
            print(f"  Mean: {rewards_at_terminals.mean():.6f}")
            print(f"  Sample (first 10): {rewards_at_terminals[:10]}")
            
            # Check if ALL non-zero rewards are at terminals
            nonzero_indices = np.where(rewards != 0)[0]
            nonzero_at_terminals = np.isin(nonzero_indices, terminal_indices)
            pct_at_terminals = 100 * np.sum(nonzero_at_terminals) / len(nonzero_indices) if len(nonzero_indices) > 0 else 0
            print(f"\nNon-zero rewards at terminal states: {np.sum(nonzero_at_terminals)}/{len(nonzero_indices)} ({pct_at_terminals:.1f}%)")
    
    print("\n" + "=" * 60)
    print("SPARSE REWARD TRANSFORMATION")
    print("=" * 60)
    sparse_rewards = (rewards != 0.0) * -1.0
    print("Code: sparse_rewards = (rewards != 0.0) * -1.0")
    print(f"\nResult:")
    print(f"  Unique values: {np.unique(sparse_rewards)}")
    print(f"  Number of -1: {np.sum(sparse_rewards == -1)}")
    print(f"  Number of 0: {np.sum(sparse_rewards == 0)}")
    
    data.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Try to find a default dataset
        default_path = "~/.ogbench/data/cube-triple-play-singletask-task2-v0-train-0.npz"
        dataset_path = os.path.expanduser(default_path)
        
        if not os.path.exists(dataset_path):
            print("Usage: python inspect_rewards.py <dataset_path>")
            print("\nExample:")
            print("  python inspect_rewards.py ~/.ogbench/data/cube-triple-play-singletask-task2-v0-train-0.npz")
            print("\nOr provide path to any .npz dataset file")
            sys.exit(1)
    
    inspect_rewards(dataset_path)
