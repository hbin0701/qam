# Dense Reward Computation - Important Clarification

## The Issue

You're absolutely right! There's a subtle but important issue with how dense rewards should be computed for action chunking.

### Current Implementation (INCORRECT for potential-based rewards)

```python
# In process_train_dataset()
for i in range(num_transitions):
    reward[i] = progress(i) - progress(i-1)
```

Then `sample_sequence()` accumulates:
```python
rewards[..., i] = rewards[..., i-1] + (reward[cur_idxs] * discount^i)
```

### The Problem

For potential-based rewards with action chunking, this **double-counts** the intermediate progress!

**Example with horizon_length=5:**
- Transition rewards: [0.1, 0.1, 0.1, 0.1, 0.2] (sum = 0.6)
- After accumulation: 0.1 + 0.1γ + 0.1γ² + 0.1γ³ + 0.2γ⁴

But the **correct** potential-based reward should be:
- **Single reward at end of chunk**: progress(t+5) - progress(t) = 0.6
- Placed only at the last step: [0, 0, 0, 0, 0.6]

## The Correct Approach

### Option 1: Compute Chunk-Level Rewards (RECOMMENDED)

Modify `process_train_dataset()` to compute rewards **per chunk**, not per transition:

```python
elif dense_wrapper is not None:
    print("Computing dense rewards for dataset (chunk-based)...")
    if 'qpos' not in ds:
        raise ValueError("Dense rewards require 'qpos' data.")
    
    qpos_data = ds['qpos']
    qvel_data = ds.get('qvel', None)
    terminals = ds['terminals']
    num_transitions = len(ds['rewards'])
    dense_rewards = np.zeros(num_transitions, dtype=np.float32)
    
    # Compute rewards per chunk
    horizon = FLAGS.horizon_length
    for i in tqdm.tqdm(range(0, num_transitions, horizon), desc="Computing chunk rewards"):
        # Find chunk boundaries (respecting episode boundaries)
        chunk_start = i
        chunk_end = min(i + horizon, num_transitions)
        
        # Check if chunk crosses episode boundary
        if np.any(terminals[chunk_start:chunk_end-1]):
            # Find actual end of chunk (before terminal)
            term_idx = np.where(terminals[chunk_start:chunk_end-1])[0][0]
            chunk_end = chunk_start + term_idx + 1
        
        if chunk_end > chunk_start:
            # Compute progress delta over entire chunk
            qpos_start = qpos_data[chunk_start]
            qpos_end = qpos_data[chunk_end - 1]
            qvel_start = qvel_data[chunk_start] if qvel_data is not None else None
            qvel_end = qvel_data[chunk_end - 1] if qvel_data is not None else None
            
            reward, _ = dense_wrapper.compute_reward(
                qpos_start, qpos_end, qvel_start, qvel_end
            )
            
            # Place entire reward at the LAST step of the chunk
            dense_rewards[chunk_end - 1] = reward
    
    # Replace rewards
    ds_dict = {k: v for k, v in ds.items()}
    ds_dict["rewards"] = dense_rewards
    ds = Dataset.create(**ds_dict)
    print(f"Chunk rewards computed. Mean: {dense_rewards.mean():.4f}")
```

### Option 2: Keep Per-Transition, Disable Accumulation

If you want to keep per-transition rewards, you need to modify `sample_sequence()` to NOT accumulate them (treat them as already-accumulated rewards).

## Which Should We Use?

**Option 1 (chunk-based)** is more correct for potential-based rewards because:
1. Matches the theoretical formulation: R(s,a) = φ(s') - φ(s)
2. Avoids double-counting intermediate progress
3. More stable learning signal

**Current implementation** (per-transition with accumulation) effectively gives:
- More frequent reward signals
- May help with credit assignment
- But theoretically incorrect for potential-based shaping

## Recommendation

Use **Option 1** for theoretically correct potential-based rewards. The current implementation may work in practice but could lead to over-optimistic value estimates.

Would you like me to implement Option 1?
