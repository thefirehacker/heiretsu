# Phase 5: Tensor Parallelism (TP)

**Week:** 4  
**Estimated Time:** 5-6 hours  
**Prerequisites:** Phase 1 (DP), understanding of matrix multiplication

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** column-parallel and row-parallel linear layers
2. **Implement** autograd-aware collective operations
3. **Debug** the SmolLM3 initialization bug and fix
4. **Trace** all-reduce patterns for activations and gradients

---

## Theory (Course Content)

### Session: Tensor Parallelism (1.5 hours)

- [ ] Watch: **Tensor Parallelism** lecture
- [ ] Understand: Why TP (split large weight matrices)
- [ ] Key concepts:
  - Column-parallel: Shard output features (each GPU computes part of output)
  - Row-parallel: Shard input features (each GPU has part of weight, all-reduce result)
  - Paired pattern: Column → Row eliminates redundant all-reduces

### Session: TensorParallelism with TorchTitan (1 hour)

- [ ] Watch: **TP with TorchTitan** lecture
- [ ] See: DTensor abstraction for cleaner TP code
- [ ] Learn: How production systems handle TP

### Guest Lectures

- [ ] Watch: **Async Tensor Parallelism** - Less Wright (Meta)
- [ ] Watch: **Introduction to TorchTitan** - Wanchao Liang

---

## Practice (Codebase)

### Step 1: Understand Autograd-Aware Collectives

**File:** `tp_linear.py` - Lines 19-61

These are the building blocks of TP. Study each:

**_CopyToModelParallelRegion (lines 19-34)**
```python
# Forward: identity (input passes through)
# Backward: all-reduce gradients (sum from all TP ranks)
```
- [ ] Read and understand: Used BEFORE column-parallel layer
- [ ] Why: Gradients flowing back need to be summed across TP ranks

**_ReduceFromModelParallelRegion (lines 37-51)**
```python
# Forward: all-reduce (sum partial results)
# Backward: identity (gradient passes through)
```
- [ ] Read and understand: Used AFTER row-parallel layer
- [ ] Why: Row-parallel computes partial sums that need combining

### Step 2: Understand ColumnParallelLinear

**File:** `tp_linear.py` - `ColumnParallelLinear` class (lines 87-167)

```
Full weight: W[out, in]
Sharded:     W_shard[out/TP, in]  # Each GPU has rows
```

- [ ] Read: `__init__` - How output dimension is sharded
- [ ] Read: `reset_parameters` - Per-rank seed for diversity (SmolLM3 fix!)
- [ ] Read: `forward` - Matrix multiply with local shard

**Key forward flow:**
```python
x = copy_to_model_parallel_region(x, self.tp_group)  # Identity fwd, all-reduce bwd
y_shard = einops.einsum(x2d, W_shard, 'bs d, op d -> bs op')  # Local matmul
# Output: y_shard (B,S,out/TP)
```

### Step 3: Understand RowParallelLinear

**File:** `tp_linear.py` - `RowParallelLinear` class (lines 245-337)

```
Full weight: W[out, in]
Sharded:     W_shard[out, in/TP]  # Each GPU has columns
```

- [ ] Read: `__init__` - How input dimension is sharded
- [ ] Read: `forward` - Partial matmul + all-reduce

**Key forward flow:**
```python
# Input is sharded: x_shard (B,S,in/TP)
y_partial = einops.einsum(x_2d, W_shard, 'bs d, op d -> bs op')  # Local matmul
y_full = reduce_from_model_parallel_region(y_partial, self.tp_group)  # All-reduce
# Output: y_full (B,S,out) - complete result
```

### Step 4: Understand the SmolLM3 Bug Fix

**File:** `tp_linear.py` - `reset_parameters()` method

This was a real bug in SmolLM3 training! Without per-rank seeds, all TP ranks had identical weight initializations.

```python
def reset_parameters(self, std: float = 0.02, base_seed: int = 1337) -> None:
    # Unique seed per TP-rank, layer, and module
    seed = base_seed + self.tp_rank * 1000000 + self.layer_idx * 100 + self.module_id
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    self.weight.normal_(mean=0.0, std=std, generator=gen)
```

- [ ] Understand: Why identical seeds caused poor convergence
- [ ] Find: Where `layer_idx` and `module_id` are passed in `gpt_model.py`

### Step 5: See TP in Model

**File:** `gpt_model.py` - `AttentionTP` and `MLPTP` classes

- [ ] Find: `ColumnParallelLinearQKV` for Q/K/V projections
- [ ] Find: `RowParallelLinear` for output projection
- [ ] Pattern: Column-parallel → Attention → Row-parallel

```python
# AttentionTP pattern:
self.c_attn = ColumnParallelLinearQKV(...)  # QKV in one layer
self.c_proj = RowParallelLinear(...)         # Output projection

# MLPTP pattern:
self.c_fc = ColumnParallelLinear(...)        # Up projection
self.c_proj = RowParallelLinear(...)         # Down projection
```

### Step 6: Run TP Tests

- [ ] Run: TP forward parity test
```bash
torchrun --standalone --nproc_per_node=2 tests/forward_tp.py --tp 2
```

- [ ] Run: TP equivalence test (forward + gradients)
```bash
torchrun --standalone --nproc_per_node=2 tests/tests_equiv.py --tp 2
```

- [ ] Run: TP comparison script
```bash
bash tests/compare_tp.sh
```

- [ ] Run: Combined DP+TP test
```bash
torchrun --standalone --nproc_per_node=4 tests/parallel_sanity.py --dp 2 --tp 2
```

---

## Concept Bridge: Column-Row Parallelism

**Theory** (from course):
> Column-parallel shards output features, row-parallel shards input features. Pairing them avoids redundant communication.

**Implementation** (in codebase):

```
Column-Parallel (W sharded by rows):
┌─────────┐     ┌─────────┐
│ x (full)│     │W0 (rows)│ → y0 (partial out)
│ x (full)│  @  │W1 (rows)│ → y1 (partial out)
└─────────┘     └─────────┘
                Output: y0 and y1 on different GPUs (concatenated conceptually)

Row-Parallel (W sharded by columns):
┌──────────┐    ┌──────────┐
│x0 (shard)│    │W0 (cols) │ → partial0
│x1 (shard)│ @  │W1 (cols) │ → partial1
└──────────┘    └──────────┘
                All-reduce: y = partial0 + partial1
```

**Connection:**
- Course teaches the sharding strategy
- Code shows exact implementation with einops
- The Column→Row pairing means:
  - Column output is sharded → directly feeds Row input (no communication!)
  - Row does all-reduce at the end

---

## TP Communication Pattern

```
Transformer Block with TP=2:

Input x (replicated on both GPUs)
         │
         ▼
┌────────────────────────────────────────────────────┐
│ ColumnParallelLinearQKV (no comm in forward)       │
│   GPU 0: computes Q0, K0, V0                       │
│   GPU 1: computes Q1, K1, V1                       │
└────────────────────────────────────────────────────┘
         │
         ▼ (sharded heads, no communication)
┌────────────────────────────────────────────────────┐
│ Attention (local computation per GPU)              │
│   Each GPU handles its subset of attention heads   │
└────────────────────────────────────────────────────┘
         │
         ▼ (sharded output)
┌────────────────────────────────────────────────────┐
│ RowParallelLinear c_proj (all-reduce in forward)   │
│   GPU 0: partial_out                               │
│   GPU 1: partial_out                               │
│   ALL-REDUCE → full output on all GPUs             │
└────────────────────────────────────────────────────┘
         │
         ▼
Output (replicated on both GPUs)
```

---

## Hands-On Exercises

### Exercise 1: Verify Weight Sharding

Add prints to confirm sharding:

```python
# In ColumnParallelLinear.__init__
print(f"TP rank {self.tp_rank}: weight shape = {self.weight.shape}")
print(f"  Full out={self.out_features}, Local out={self.out_per_partition}")
```

### Exercise 2: Trace All-Reduce

Add timing to measure communication overhead:

```python
import time
start = time.time()
dist.all_reduce(y_partial, op=dist.ReduceOp.SUM, group=self.tp_group)
print(f"All-reduce took {(time.time() - start)*1000:.2f}ms")
```

### Exercise 3: Break TP Init (Reproduce SmolLM3 Bug)

1. Modify `reset_parameters` to use same seed on all ranks
2. Train for 500 steps
3. Compare loss curve with proper per-rank seeds
4. You should see slower convergence or degraded performance

---

## Verification Checklist

- [ ] Can explain: "Column-parallel shards output, row-parallel shards input"
- [ ] Can trace: Where all-reduce happens (RowParallelLinear forward)
- [ ] Can trace: Where gradients are all-reduced (CopyToModelParallelRegion backward)
- [ ] Understand: Why per-rank seeds matter for TP initialization
- [ ] Successfully ran `tests_equiv.py` with diff < 1e-5
- [ ] Know: TP is best for reducing per-layer memory, not total memory

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Shape mismatch | out_features not divisible by TP | Ensure model dims are TP-divisible |
| High gradient diff | Numerical precision | Use fp32 for testing, check order of ops |
| Poor convergence | Same seeds across TP ranks | Use per-rank initialization seeds |
| Slow training | Too much all-reduce | Batch multiple layers, use async TP |

---

## Next Phase

Now you understand how to shard layers (TP). With MoE, we have many experts. Expert Parallelism distributes those experts across GPUs, complementing TP.

[Continue to Phase 6: Expert Parallelism →](06-Phase-EP.md)
