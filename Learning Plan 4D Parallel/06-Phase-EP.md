# Phase 6: Expert Parallelism (EP)

**Week:** Bonus  
**Estimated Time:** 4-5 hours  
**Prerequisites:** Phase 4 (MoE), Phase 1 (DP for understanding groups)

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** how EP distributes experts across GPUs
2. **Implement** All-to-All communication for token dispatch
3. **Trace** the dispatch-compute-combine pattern
4. **Combine** EP with DP and TP (4D parallelism)

---

## Theory (Course Content)

### Guest Lecture: Expert Parallelism (Essential)

- [ ] Watch: **Expert Parallelism** - Matej Sirovatka (Hugging Face)
- [ ] Understand: Why EP (too many experts for one GPU)
- [ ] Key concepts:
  - Each GPU owns a subset of experts (E_local = E / EP_world)
  - All-to-All sends tokens to GPU owning target expert
  - All-to-All receives processed tokens back

### Related Lectures

- [ ] Watch: **Multi-dimensional parallelism** - Ferdinand Mom (Hugging Face)
- [ ] See how EP combines with DP, TP, PP

---

## Practice (Codebase)

### Step 1: Understand EP Groups

**File:** `topo.py` - EP group creation (lines 67-73)

```python
# EP groups: same dp, pp, tp; varying ep
for d in range(dp):
    for p in range(pp):
        for t in range(tp):
            ranks = [rank_from_coords(d, e, p, t, ...) for e in range(ep)]
```

- [ ] Understand: EP group contains ranks that share experts
- [ ] Verify: All ranks in EP group have same (dp, pp, tp) coordinates

### Step 2: Understand All-to-All Dispatch

**File:** `ep_comm.py` - `all_to_all_dispatch()` function

```python
def all_to_all_dispatch(
    x: torch.Tensor,           # Tokens to send
    send_counts: torch.Tensor, # How many tokens to each EP rank
    ep_group: ProcessGroup,
    ep_world_size: int,
    payload: torch.Tensor,     # Expert indices (sent alongside tokens)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

- [ ] Read: How `send_counts` determines split
- [ ] Understand: `dist.all_to_all_single` exchanges tokens
- [ ] Note: Payload (expert indices) is sent with tokens

### Step 3: Understand All-to-All Combine

**File:** `ep_comm.py` - `all_to_all_combine()` function

```python
def all_to_all_combine(
    y: torch.Tensor,           # Processed tokens
    send_counts: torch.Tensor, # Original send counts
    recv_counts: torch.Tensor, # What we received (now we send back)
    ep_group: ProcessGroup,
    ep_world_size: int,
) -> torch.Tensor:
```

- [ ] Understand: Reverses the dispatch to return tokens to origin
- [ ] Note: Uses recv_counts from dispatch as send_counts for combine

### Step 4: Trace EP in MoE Forward

**File:** `moe.py` - EP-enabled path in `MoELayer.forward()` (lines 343-380)

Study the EP code path:

**Step A: Determine destination EP rank**
```python
experts_per_rank = self.num_local_experts
ep_indices = indices_sorted // experts_per_rank  # Which EP rank owns each expert
send_counts = torch.bincount(ep_indices, minlength=self.ep_world_size)
```

**Step B: Dispatch tokens to EP ranks**
```python
x_recv, recv_counts, idx_recv = all_to_all_dispatch(
    x_for_a2a, send_counts, self.ep_group, self.ep_world_size, payload=idx_for_a2a
)
```

**Step C: Process local experts**
```python
local_start = self.ep_rank * self.num_local_experts
local_indices = idx_recv - local_start  # Map to local expert IDs [0, E_local)
y_local_sorted = self.expert_group(x_local_sorted, local_counts, local_indices_sorted)
```

**Step D: Combine (return tokens to origin)**
```python
y_return = all_to_all_combine(
    y_local, send_counts, recv_counts, self.ep_group, self.ep_world_size
)
```

### Step 5: Run EP Tests

- [ ] Run: Full test suite with EP configs (requires 8 GPUs)
```bash
bash tests/run_full_suite.sh
```

EP-specific configs in the test suite:
- `dp=2 ep=2 tp=2 pp=1` (4D MoE)
- `dp=4 ep=2 tp=1 pp=1` (DP + EP)
- `dp=1 ep=2 tp=2 pp=2` (EP + TP + PP)

- [ ] If you have 4 GPUs, run manually:
```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --dp 2 --ep 2 --tp 1 --pp 1 \
    --num_experts 8 --top_k 2 --moe_freq 2 \
    --max_iters 100
```

---

## Concept Bridge: All-to-All Communication

**Theory** (from guest lecture):
> All-to-All allows each GPU to send different amounts to each other GPU. For EP, tokens are sent to the GPU that owns the target expert.

**Implementation** (in codebase):

```python
# ep_comm.py - All-to-All dispatch
# send_counts: [10, 5, 8, 7]  # Tokens to send to each EP rank
# recv_counts: [12, 6, 9, 3]  # Tokens to receive from each EP rank

# Each EP rank gets different tokens
output_splits = list(recv_counts.cpu().tolist())
input_splits = list(send_counts.cpu().tolist())

y_recv = torch.empty(sum(output_splits), H, device=x.device)
dist.all_to_all_single(y_recv, x, output_split_sizes=output_splits, 
                       input_split_sizes=input_splits, group=ep_group)
```

**Connection:**
- Course teaches All-to-All concept
- Code shows exactly how variable-size exchanges work
- Key insight: Token distribution is dynamic based on routing

---

## EP Communication Pattern

```
4 GPUs with EP=4, 8 experts (2 per GPU):

Initial state (after routing):
┌────────────────────────────────────────────────────────────────────┐
│ GPU 0                  GPU 1                  GPU 2       GPU 3    │
│ tokens for E0,E1,E5   tokens for E2,E7       tokens...   tokens... │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ All-to-All Dispatch
┌────────────────────────────────────────────────────────────────────┐
│ GPU 0: owns E0,E1     GPU 1: owns E2,E3     GPU 2: E4,E5  GPU 3: E6,E7│
│ receives tokens       receives tokens        receives    receives   │
│ for E0,E1 from all    for E2,E3 from all    tokens...   tokens...  │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ Local Expert Processing
┌────────────────────────────────────────────────────────────────────┐
│ GPU 0: Expert(E0), Expert(E1) process their tokens                 │
│ GPU 1: Expert(E2), Expert(E3) process their tokens                 │
│ ... etc                                                            │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ All-to-All Combine
┌────────────────────────────────────────────────────────────────────┐
│ GPU 0: gets back its original tokens (now processed)               │
│ GPU 1: gets back its original tokens (now processed)               │
│ ... etc                                                            │
└────────────────────────────────────────────────────────────────────┘
```

---

## Hands-On Exercises

### Exercise 1: Trace Token Movement

Add logging to see token counts:

```python
# In MoELayer.forward(), EP path
print(f"EP rank {self.ep_rank}: send_counts = {send_counts.tolist()}")
print(f"EP rank {self.ep_rank}: recv_counts = {recv_counts.tolist()}")
print(f"EP rank {self.ep_rank}: processing {x_recv.shape[0]} tokens locally")
```

### Exercise 2: Verify Expert Ownership

Check that each GPU processes the right experts:

```python
# After computing local_indices
unique_experts = local_indices.unique().tolist()
expected = list(range(self.num_local_experts))
assert unique_experts == expected or len(unique_experts) == 0, \
    f"EP rank {self.ep_rank} got wrong experts: {unique_experts}"
```

### Exercise 3: Measure All-to-All Latency

```python
import time
torch.cuda.synchronize()
start = time.time()
x_recv, recv_counts, idx_recv = all_to_all_dispatch(...)
torch.cuda.synchronize()
print(f"All-to-All dispatch: {(time.time() - start)*1000:.2f}ms")
```

---

## Verification Checklist

- [ ] Can explain: "EP distributes experts across GPUs, uses All-to-All"
- [ ] Can trace: Token flow through dispatch → process → combine
- [ ] Understand: send_counts and recv_counts relationship
- [ ] Understand: How EP group differs from DP/TP/PP groups
- [ ] Know: EP is best for scaling number of experts, not expert size

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| All-to-All hang | Mismatched counts across ranks | Verify send/recv counts sum correctly |
| Wrong expert processing | Incorrect local index mapping | Check `local_start` calculation |
| Token loss | Asymmetric dispatch/combine | Verify inverse permutation |
| OOM | Uneven token distribution | Add capacity factor or drop tokens |

---

## Next Phase

Now you understand all 4 dimensions: DP, TP, PP, and EP. The final phase integrates them into 4D parallelism.

[Continue to Phase 7: 4D Integration →](07-Phase-Integration.md)
