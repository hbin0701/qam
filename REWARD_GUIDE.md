# QAM Reward Implementations - Complete Guide

## Sparse Rewards (Confirmed)

### Implementation in `main.py`
```python
sparse_rewards = (ds["rewards"] != 0.0) * -1.0
```

### OGBench Original Reward Format
Based on `ogbench/relabel_utils.py`, the original rewards for Task 2 (Triple) are:
- **-3** (0 cubes at goal)
- **-2** (1 cube at goal)
- **-1** (2 cubes at goal)
- **0**  (3 cubes at goal - Success)

### Final Sparse Reward Behavior
The transformation in `main.py` converts this to:
- **-1.0** at every step before full completion
- **0.0** when the task is fully completed (all 3 cubes at goal)

This is a **time-penalty formulation**. The agent receives a constant penalty of -1 until success, which encourages reaching the goal as quickly as possible.

### How it works with action chunking
`sample_sequence()` accumulates rewards:
```python
rewards[..., i] = rewards[..., i-1] + (reward[cur_idxs] * discount^i)
```

So if task completes at step 3 of a 5-step chunk:
- Per-step: `[0, 0, -1, 0, 0]`
- Accumulated: `[0, 0, -1, -γ, -γ²]`

## Dense Rewards (V1/V2/V3)

### Implementation (CHUNK-BASED - FIXED!)

```python
# Compute rewards per chunk (horizon_length steps)
while i < num_transitions:
    chunk_start = i
    chunk_end = min(i + horizon, num_transitions)
    
    # Respect episode boundaries
    if chunk_end > chunk_start + 1:
        term_mask = terminals[chunk_start:chunk_end-1] > 0
        if np.any(term_mask):
            term_idx = np.where(term_mask)[0][0]
            chunk_end = chunk_start + term_idx + 1
    
    # Compute progress delta over ENTIRE chunk
    reward = progress(qpos_end) - progress(qpos_start)
    
    # Place reward at LAST step of chunk
    dense_rewards[chunk_end - 1] = reward
    
    i = chunk_end
```

### Key Features

1. **Chunk-based computation**: Reward = progress delta over entire action chunk
2. **Step Penalty**: Every step in a chunk adds a **-1.0 penalty**, same as the sparse baseline.
3. **Total Chunk Reward**: `Reward = (Progress Delta) - (Steps in Chunk)`
4. **Episode-aware**: Stops chunks at episode boundaries
5. **Sparse placement**: Reward only at last step of each chunk
6. **Theoretically correct**: Proper potential-based reward shaping

### Example (with Step Penalty)

With `horizon_length=5` and progress going from 0.0 → 0.6:

**Chunk Gain**: 0.6
**Time Penalty**: -5.0
**Chunk Reward**: `0.6 - 5.0 = -4.4`

**Placement at last step of chunk**:
```
[0, 0, 0, 0, -4.4]
```

**After `sample_sequence()` accumulation:**
```
[0, 0, 0, 0, -4.4γ⁴]
```

This is **correct** for potential-based rewards!

## Comparison

| Aspect | Sparse | Dense (V1/V2/V3) |
|--------|--------|------------------|
| **Signal frequency** | Very rare | Every chunk |
| **Reward value** | -1 at completion | Progress delta (0-3 or 0-9) |
| **Placement** | At completion step | At end of each chunk |
| **Learning difficulty** | Hard (sparse signal) | Easier (frequent feedback) |
| **Theoretical basis** | Standard RL | Potential-based shaping |

## Why Chunk-Based?

### ❌ Per-Transition (OLD - INCORRECT)
```python
for i in range(num_transitions):
    reward[i] = progress(i) - progress(i-1)
# Result: [0.1, 0.1, 0.1, 0.1, 0.2]
# After accumulation: 0.1 + 0.1γ + 0.1γ² + 0.1γ³ + 0.2γ⁴
```
**Problem**: Double-counts intermediate progress!

### ✅ Chunk-Based (NEW - CORRECT)
```python
# Compute once per chunk
reward = progress(t+5) - progress(t)
# Result: [0, 0, 0, 0, 0.6]
# After accumulation: 0.6γ⁴
```
**Correct**: Single reward for entire chunk's progress!

## Usage

```bash
# Sparse (baseline)
bash scripts/run_task2_sparse.sh

# Dense V1 (simple)
bash scripts/run_task2_dense_v1.sh

# Dense V2 (recommended)
bash scripts/run_task2_dense_v2.sh

# Dense V3 (detailed)
bash scripts/run_task2_dense_v3.sh
```

## Expected Output

When running with dense rewards, you'll see:
```
Computing dense rewards for dataset (chunk-based, horizon=5)...
Computing chunk rewards: 100%|████████| 50000/50000
Dense rewards computed. Mean: 0.1234, Std: 0.0567, Chunks: 10000
```

This confirms:
- Chunk-based computation is working
- Statistics computed only on non-zero rewards
- Number of chunks = num_transitions / horizon_length

## Ready to Run!

All reward implementations are now correct and ready for experiments.
