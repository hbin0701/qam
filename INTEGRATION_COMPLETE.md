# QAM Dense Reward Integration - Complete

## Summary

Successfully integrated dense reward computation into QAM's training pipeline. The system now supports 4 reward modes:
- **Sparse** (original): Binary reward on task completion
- **Dense V1**: Simple cube placement count
- **Dense V2**: Distance-based progress (recommended)
- **Dense V3**: Detailed stage tracking

## Changes Made to `main.py`

### 1. Added Import
```python
from envs.rewards.cube_triple_play_task2 import DenseRewardWrapper
```

### 2. Added Flag
```python
flags.DEFINE_string('dense_reward_version', None, 'Dense reward version (v1/v2/v3), None for original rewards')
```

### 3. Initialize Wrapper
After config setup (line ~110):
```python
dense_wrapper = None
if FLAGS.dense_reward_version is not None:
    print(f"Using dense reward version: {FLAGS.dense_reward_version}")
    dense_wrapper = DenseRewardWrapper(
        task_name=FLAGS.env_name,
        version=FLAGS.dense_reward_version,
        debug=False
    )
```

### 4. Modified `process_train_dataset()`
Added dense reward computation:
- Loads qpos/qvel data from dataset
- Computes rewards as progress deltas: `reward[i] = progress(i) - progress(i-1)`
- Replaces original rewards with dense rewards
- Shows progress bar and statistics

### 5. Updated Dataset Loading
Added `add_info=True` to all `make_ogbench_env_and_datasets` calls when using dense rewards:
- Initial dataset load (line ~97)
- Offline phase dataset replacement (line ~264)
- Online phase dataset replacement (line ~330)

## Usage

Run the provided scripts:

```bash
cd /rlwrld3/home/hyeonbin/RL/qam

# Sparse (baseline)
bash scripts/run_task2_sparse.sh

# Dense V1
bash scripts/run_task2_dense_v1.sh

# Dense V2 (recommended)
bash scripts/run_task2_dense_v2.sh

# Dense V3 (most detailed)
bash scripts/run_task2_dense_v3.sh
```

## How It Works

1. **Dataset Loading**: When `dense_reward_version` is set, loads qpos/qvel data via `add_info=True`
2. **Reward Computation**: In `process_train_dataset()`, computes dense rewards for each transition
3. **Progress Tracking**: Uses physics-based state to compute progress (0-3 for V1/V2, 0-9 for V3)
4. **Reward Shaping**: Reward = progress delta (potential-based, zero-sum over trajectories)

## Performance Notes

- Dense reward computation adds ~1-2 minutes to dataset loading
- Progress bar shows computation progress
- Statistics (mean/std) printed after computation
- Rewards cached in dataset, no runtime overhead during training

## Files Modified

- ✅ `main.py` - Core integration
- ✅ `scripts/run_task2_sparse.sh` - Sparse baseline
- ✅ `scripts/run_task2_dense_v1.sh` - Dense V1
- ✅ `scripts/run_task2_dense_v2.sh` - Dense V2
- ✅ `scripts/run_task2_dense_v3.sh` - Dense V3

## Ready to Run!

All integration complete. Scripts are ready to use for experiments comparing different reward models.
