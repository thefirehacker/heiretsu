# Phase 1: Data Parallelism (DP)

**Week:** 1  
**Estimated Time:** 3-4 hours  
**Prerequisites:** PyTorch basics, understanding of gradients

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** how data parallelism splits batches across GPUs
2. **Implement** gradient averaging with `all_reduce`
3. **Verify** that DP produces identical results to single-GPU training
4. **Debug** common DP issues (grad sync, determinism)

---

## Theory (Course Content)

### Session: Distributed Data Parallelism From Scratch (1 hour)

- [ ] Watch: **DDP from Scratch** lecture
- [ ] Understand: Why DDP is the foundation of all distributed training
- [ ] Key concepts:
  - Batch sharding: `B_global = B_local * world_size`
  - Gradient averaging: `grad_avg = Σ grads / world_size`
  - Identical models on all ranks (no weight sharding)

### Session: DataLoader Workshop

- [ ] Watch: **DataLoader Workshop**
- [ ] Understand: `DistributedSampler` and avoiding data overlap
- [ ] Key insight: Each rank sees different data, but all have same model

---

## Practice (Codebase)

### Step 1: Read the Implementation

**File:** `dp.py` (32 lines - read entirely)

```python
# Key function to understand:
def average_gradients(model, dp_group):
    """All-reduce grads across DP group (SUM then /world_size)."""
```

- [ ] Read: `dp.py` completely
- [ ] Trace the gradient flow:
  1. Each rank computes local gradients (backward pass)
  2. `all_reduce` sums gradients across ranks
  3. Divide by world size to get average
- [ ] Note: The MoE handling for unused experts (lines 27-29)

**Key einops hint from code:**
```
Batch dim sharding: b -> (dp b_local)
```

### Step 2: Understand Topology

**File:** `topo.py` - Focus on DP groups

- [ ] Read: `init_topology()` function
- [ ] Find: How `dp_group` is created in `build_groups()`
- [ ] Understand: DP ranks share same (ep, pp, tp) coordinates

```python
# DP groups: same ep, pp, tp; varying dp
for e in range(ep):
    for p in range(pp):
        for t in range(tp):
            ranks = [rank_from_coords(d, e, p, t, ...) for d in range(dp)]
```

### Step 3: Run DP Tests

- [ ] Run: Single-GPU baseline
```bash
python tests/forward_single.py
```

- [ ] Run: 2-GPU DP forward test
```bash
torchrun --standalone --nproc_per_node=2 tests/forward_dp.py
```

**Expected output:**
```
DP parity diff_max 0.0 diff_mean 0.0
```
This confirms all ranks produce identical logits.

- [ ] Run: DP comparison script
```bash
bash tests/compare_dp.sh
```

### Step 4: Trace Training Loop

**File:** `train.py` - Search for DP usage

- [ ] Find: Where `average_gradients` is called
- [ ] Find: How `topo.dp_group` is passed
- [ ] Find: How batch data is split across DP ranks (look for `dp_rank` in data loading)

---

## Concept Bridge: Gradient Averaging

**Theory** (from course):
> In DDP, each GPU processes a different mini-batch. After backward pass, we average gradients across all GPUs so they stay synchronized.

**Implementation** (in codebase):

```python
# dp.py lines 13-31
def average_gradients(model: torch.nn.Module, dp_group) -> None:
    world = dist.get_world_size(group=dp_group)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)  # MoE: unused experts
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
        p.grad.div_(world)  # Average = SUM / world_size
```

**Connection:**
- Course teaches the concept of gradient averaging
- Code shows the exact `all_reduce` + divide implementation
- Notice: MoE experts may have `None` grads (unused on some batches)

---

## Hands-On Exercises

### Exercise 1: Verify Gradient Sync

Modify `tests/forward_dp.py` to also check gradient equality:

1. Add a backward pass after forward
2. Compare gradients across ranks using `all_gather`
3. Assert gradients are identical

### Exercise 2: Break DP

Intentionally break DP and observe failures:

1. Comment out `average_gradients` call in `train.py`
2. Run training for 100 steps
3. Compare weights across ranks - they should diverge!

### Exercise 3: Manual Gradient Check

Add debug prints to `average_gradients`:

```python
print(f"Rank {dist.get_rank()}: grad norm before = {p.grad.norm()}")
dist.all_reduce(...)
print(f"Rank {dist.get_rank()}: grad norm after = {p.grad.norm()}")
```

---

## Verification Checklist

- [ ] Can explain in own words: "DP splits batches, averages gradients"
- [ ] Can trace the gradient flow through `dp.py`
- [ ] Successfully ran `forward_dp.py` with diff=0
- [ ] Understand why all ranks must have identical weights
- [ ] Know the formula: `effective_batch = batch_per_gpu * dp_world_size`

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `diff_max > 0` | Different random seeds | Set same seed on all ranks |
| `NCCL timeout` | One rank has `grad=None` | Initialize zero grad (see MoE handling) |
| OOM on some ranks | Uneven batch sizes | Use DistributedSampler with drop_last=True |

---

## Next Phase

After mastering DP, you understand the foundation. Next:
- **Phase 2: ZeRO** - What if we want to reduce memory further?
- **Phase 3: PP** - What if one model doesn't fit on one GPU?

[Continue to Phase 2: ZeRO →](02-Phase-ZeRO.md)
