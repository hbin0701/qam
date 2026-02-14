# Dense Reward Implementation - QAM

## Summary

Implemented **three versions** of dense reward models for OGBench Task 2 (Triple Pick and Place):

### Version Overview

| Version | Files | Progress Range | Complexity |
|---------|-------|----------------|------------|
| **V1** | `dense_v1.py` | 0.0 - 3.0 | Simple cube count |
| **V2** | `dense_v2.py` | 0.0 - 3.0 | Distance-based |
| **V3** | `dense_v3.py` | 0.0 - 12.0 | Detailed stages |

### Quick Start

```python
from envs.rewards.cube_triple_play_task2 import DenseRewardWrapper

# Choose version: "v1", "v2", or "v3"
wrapper = DenseRewardWrapper(version="v2")
reward, info = wrapper.compute_reward(qpos_prev, qpos_curr)
```

### Files Created

```
envs/rewards/cube-triple-play-task2/
├── __init__.py           # Package exports
├── wrapper.py            # Unified wrapper (supports all versions)
├── dense_v1.py          # V1: Simple cube placement count
├── dense_v2.py          # V2: Distance-based progress
├── dense_v3.py          # V3: Detailed stage tracking
├── README.md            # Version comparison & usage guide
└── progress.py          # (old file, can be removed)
```

### Recommendation

**Start with V2** - best balance of informativeness and simplicity.

See [`envs/rewards/cube-triple-play-task2/README.md`](file:///rlwrld3/home/hyeonbin/RL/qam/envs/rewards/cube-triple-play-task2/README.md) for detailed comparison.

## Testing

All versions tested and working:
```bash
cd envs/rewards/cube-triple-play-task2
python3 wrapper.py
```

✅ V1, V2, V3 all pass tests.
