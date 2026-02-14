# Dense Reward Models - Version Comparison

## Overview

Three versions of dense reward models for OGBench Task 2 (Triple Pick and Place), each with different complexity and granularity.

## Version Comparison

| Version | Complexity | Progress Range | Stage Definition | Use Case |
|---------|-----------|----------------|------------------|----------|
| **V1** | Simple | 0.0 - 3.0 | # cubes at goal | Baseline, sparse-like |
| **V2** | Medium | 0.0 - 3.0 | # cubes at goal + distance | Smooth learning signal |
| **V3** | Detailed | 0.0 - 12.0 | Reach→Grasp→Carry→Place | Fine-grained control |

## V1: Simple Cube Placement Count

**Concept:** Reward only when cubes reach their goals.

**Stage:** Number of cubes at goal positions (0-3)

**Progress:** Integer count of completed cubes

**Pros:**
- Simple and interpretable
- Similar to sparse reward but with intermediate milestones
- Fast computation

**Cons:**
- No reward for partial progress
- May be too sparse for learning

**Example:**
```python
wrapper = DenseRewardWrapper(version="v1")
# Progress: 0.0 → 1.0 → 2.0 → 3.0
# Only changes when a cube reaches its goal
```

## V2: Distance-Based Progress

**Concept:** Reward based on how close cubes are to their goals.

**Stage:** Number of cubes at goal (0-3)

**Progress:** Completed cubes + average progress of incomplete cubes

**Calculation:**
```
progress = num_completed + (sum(incomplete_progress) / num_incomplete)
where incomplete_progress = (initial_dist - current_dist) / initial_dist
```

**Pros:**
- Smooth reward signal
- Encourages moving cubes toward goals
- Good balance of simplicity and informativeness

**Cons:**
- Doesn't distinguish between different manipulation phases
- May reward inefficient paths

**Example:**
```python
wrapper = DenseRewardWrapper(version="v2")
# Progress: 0.0 → 0.3 → 0.7 → 1.0 → 1.2 → ... → 3.0
# Smooth progression as cubes move toward goals
```

## V3: Detailed Stage Tracking

**Concept:** Track each manipulation phase with strict cube ordering.

**Stage:** `cube_idx * 4 + substage` (0-12)
- Substages: 0=Reach, 1=Grasp, 2=Carry, 3=Place

**Progress:** Fine-grained tracking through all 12 substages

**Strict Ordering:** Must complete Cube 0 before Cube 1, etc.

**Pros:**
- Most informative reward signal
- Enforces proper manipulation sequence
- Best for learning complex behaviors

**Cons:**
- More complex computation
- Strict ordering may be too constraining
- Requires accurate state estimation

**Example:**
```python
wrapper = DenseRewardWrapper(version="v3")
# Progress: 0.0 → 1.0 → 2.0 → 3.0 → 4.0 → ... → 12.0
# Each integer = completing a substage
# Cube 0: Reach(0-1) → Grasp(1-2) → Carry(2-3) → Place(3-4)
# Cube 1: Reach(4-5) → Grasp(5-6) → Carry(6-7) → Place(7-8)
# Cube 2: Reach(8-9) → Grasp(9-10) → Carry(10-11) → Place(11-12)
```

## Usage

```python
from envs.rewards.cube_triple_play_task2 import DenseRewardWrapper

# Choose version
wrapper = DenseRewardWrapper(
    task_name="cube-triple-play-singletask-task2-v0",
    version="v2",  # or "v1", "v3"
    debug=False
)

# Compute reward
reward, info = wrapper.compute_reward(qpos_prev, qpos_curr)
```

## Recommendation

**For initial experiments:** Start with **V2**
- Good balance of informativeness and simplicity
- Smooth learning signal without being too complex
- Works well with most RL algorithms

**For baseline comparison:** Use **V1**
- Similar to sparse reward
- Easy to interpret
- Good for ablation studies

**For fine-grained control:** Use **V3**
- Best for learning precise manipulation
- Useful when you need detailed progress tracking
- May require more tuning

## Integration with QAM

Add to `main.py`:
```python
flags.DEFINE_string('dense_reward_version', 'v2', 'Dense reward version (v1/v2/v3)')
```

In training loop:
```python
if FLAGS.use_dense_reward:
    from envs.rewards.cube_triple_play_task2 import DenseRewardWrapper
    reward_wrapper = DenseRewardWrapper(
        task_name=FLAGS.env_name,
        version=FLAGS.dense_reward_version
    )
```
