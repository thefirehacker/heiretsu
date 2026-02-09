# Phase 3: Pipeline Parallelism (PP)

**Week:** 3  
**Estimated Time:** 5-6 hours  
**Prerequisites:** Phase 1 (DP), Phase 2 (ZeRO concepts)

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** how PP splits model layers across GPUs
2. **Implement** GPipe-style forward-backward with microbatching
3. **Trace** P2P send/recv communication patterns
4. **Calculate** pipeline bubble overhead

---

## Theory (Course Content)

### Session: Pipeline Parallelism (1.5 hours)

- [ ] Watch: **Pipeline Parallelism** lecture
- [ ] Understand: Why PP exists (model too big for one GPU, even with ZeRO)
- [ ] Key concepts:
  - Stage: A subset of consecutive layers
  - Microbatch: Smaller batches to fill the pipeline
  - Bubble: Idle time at pipeline start/end

### Session: Pipeline Parallelism with torchtitan (1 hour)

- [ ] Watch: **PP with torchtitan** lecture
- [ ] See production-grade PP implementation
- [ ] Understand: Schedule types (GPipe, 1F1B, Interleaved)

---

## Practice (Codebase)

### Step 1: Understand Layer Partitioning

**File:** `pipeline.py` - Start with `_partition_layers()`

```python
def _partition_layers(n_layer: int, pp: int) -> List[Tuple[int, int]]:
    """Split n_layer layers across pp stages."""
```

- [ ] Read: Lines 19-33
- [ ] Trace: How 12 layers are split across 4 stages → [(0,3), (3,6), (6,9), (9,12)]
- [ ] Note: Handles uneven splits (remainder goes to earlier stages)

### Step 2: Understand StageModule

**File:** `pipeline.py` - `StageModule` class (lines 36-144)

- [ ] Read: `__init__` - How each stage gets its layers
- [ ] Find: Where embedding is added (only first stage)
- [ ] Find: Where LM head is added (only last stage)
- [ ] Note: Weight tying only works when pp=1

```python
# Key attributes:
self.is_first = stage_idx == 0        # Has embeddings
self.is_last = stage_idx == (num_stages - 1)  # Has LM head
self.blocks = ...  # Only this stage's layers
```

### Step 3: Understand P2P Communication

**File:** `pipeline.py` - `GPipeEngine._p2p()` method

- [ ] Read: Lines 178-191
- [ ] Understand: `isend` and `irecv` with tags
- [ ] Key: Tags prevent message mixup between microbatches

```python
def _p2p(self, send_tensor, recv_tensor, peer, tag):
    ops = []
    if send_tensor is not None:
        ops.append(dist.P2POp(dist.isend, send_tensor, peer, tag=tag))
    if recv_tensor is not None:
        ops.append(dist.P2POp(dist.irecv, recv_tensor, peer, tag=tag))
    dist.batch_isend_irecv(ops)
```

### Step 4: Trace Forward-Backward Schedule

**File:** `pipeline.py` - `forward_backward()` method (lines 199-294)

This is the core PP logic. Study it in three parts:

**Part A: Forward Pass (lines 222-256)**
- [ ] Trace: How microbatches flow through stages
- [ ] Find: Where activations are sent to next stage
- [ ] Note: `boundaries` list stores activations for backward

**Part B: Backward Pass (lines 264-288)**
- [ ] Trace: Reversed microbatch order
- [ ] Find: Where gradients are sent back to previous stage
- [ ] Note: `recv_inputs[m].grad` is the boundary gradient

**Part C: MoE Support**
- [ ] Find: How `aux_loss` is accumulated across stages
- [ ] Note: `state.aux_losses` collects per-microbatch aux losses

### Step 5: Run PP Tests

- [ ] Run: Single-GPU baseline first
```bash
python tests/forward_single.py
```

- [ ] Run: 2-GPU PP forward test
```bash
torchrun --standalone --nproc_per_node=2 tests/forward_pp.py --pp 2
```

- [ ] Run: PP comparison script
```bash
bash tests/compare_pp.sh
```

- [ ] Run: Multi-config sanity test
```bash
torchrun --standalone --nproc_per_node=4 tests/parallel_sanity.py --dp 1 --tp 1 --pp 4
```

---

## Concept Bridge: Microbatching

**Theory** (from course):
> Microbatching fills the pipeline to reduce bubble overhead. With M microbatches and P stages, bubble fraction = (P-1)/M.

**Implementation** (in codebase):

```python
# pipeline.py forward pass loop
for m in range(micro_batches):
    # Each microbatch flows through all stages
    if not self.is_first:
        recv_x = torch.empty((B, S, H), ...)
        self._p2p(None, recv_x, self.prev_rank, self._fwd_tag(m))
    
    # Forward through local blocks
    x, aux_loss = self.stage.forward_blocks(x)
    
    if not self.is_last:
        self._p2p(boundary.detach(), None, self.next_rank, self._fwd_tag(m))
```

**Connection:**
- Course teaches bubble overhead formula
- Code shows the actual microbatch loop with P2P communication
- `grad_accum_steps` in training acts as microbatch count

---

## Pipeline Schedule Visualization

```
GPipe Schedule (4 stages, 4 microbatches):

Stage 0: [F0][F1][F2][F3]         [B3][B2][B1][B0]
Stage 1:    [F0][F1][F2][F3]   [B3][B2][B1][B0]
Stage 2:       [F0][F1][F2][F3][B3][B2][B1][B0]
Stage 3:          [F0][F1][F2][F3][B3][B2][B1][B0]
         ┌──────────┴──────────┐└────────┴────────┘
         │    Forward Fill     ││   Backward Drain  │
         └─────────────────────┘└───────────────────┘
         
Bubble = 3 idle slots at start + 3 at end = 6 slots
Total slots = 4 stages × 8 steps = 32
Bubble fraction = 6/32 = 18.75%
```

---

## Hands-On Exercises

### Exercise 1: Trace Boundary Tensors

Add debug prints to see activation shapes being passed:

```python
# In GPipeEngine.forward_backward(), after forward block
if not self.is_last:
    print(f"Stage {self.pp_rank} sending boundary shape {boundary.shape} to stage {self.pp_rank + 1}")
```

Run with `--pp 4` and observe the flow.

### Exercise 2: Verify Gradient Flow

Add gradient norm tracking:

```python
# After backward pass
for name, p in self.stage.named_parameters():
    if p.grad is not None:
        print(f"Stage {self.pp_rank} {name} grad norm: {p.grad.norm():.4f}")
```

### Exercise 3: Bubble Calculation

With 4 stages and 8 microbatches:
1. Calculate theoretical bubble fraction
2. Add timing to measure actual idle time
3. Compare theory vs practice

---

## Verification Checklist

- [ ] Can explain: "PP splits layers, uses P2P send/recv"
- [ ] Can trace: Forward pass through 4 stages with 2 microbatches
- [ ] Can trace: Backward gradient flow in reverse order
- [ ] Successfully ran `forward_pp.py` with matching loss
- [ ] Know: Bubble formula = (P-1)/M where P=stages, M=microbatches
- [ ] Understand: Why `grad_accum_steps` matters for PP

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Deadlock | Mismatched send/recv | Check tag numbering, stage ordering |
| Shape mismatch | Wrong hidden size in recv buffer | Match `(B, S, H)` exactly |
| Gradient None | Missing backward through boundary | Ensure `requires_grad_(True)` on recv |
| OOM | Too many microbatches stored | Reduce microbatch count or use activation checkpointing |

---

## Next Phase

Now you can split layers across GPUs. But what about making individual layers smaller? That's Tensor Parallelism. But first, let's understand MoE which adds a different dimension to model scaling.

[Continue to Phase 4: Mixture of Experts →](04-Phase-MoE.md)
