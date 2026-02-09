# Phase 4: Mixture of Experts (MoE)

**Week:** 3  
**Estimated Time:** 4-5 hours  
**Prerequisites:** Phase 1 (DP), Phase 3 (PP basics)

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** token routing and top-K expert selection
2. **Implement** load balancing auxiliary loss
3. **Trace** the permutation-based dispatch pattern
4. **Debug** common MoE issues (load imbalance, unused experts)

---

## Theory (Course Content)

### Session: How to train your small MoE (1 hour)

- [ ] Watch: **How to train your small MoE** lecture
- [ ] Understand: Why MoE (conditional computation, more params, same FLOPs)
- [ ] Key concepts:
  - Router: Learns which expert handles which tokens
  - Top-K: Each token goes to K experts (typically 2)
  - Load balancing: Penalty for uneven expert usage

### Guest Lecture: Hyper-optimizing LLMs with MoE, MLA

- [ ] Watch: **Elie Bakouch** (Hugging Face) lecture
- [ ] Learn: Advanced MoE techniques and optimizations

---

## Practice (Codebase)

### Step 1: Understand the Router

**File:** `moe.py` - `Router` class (lines 34-77)

```python
class Router(nn.Module):
    """Top-K gating network for token-to-expert routing."""
```

- [ ] Read: The `forward()` method
- [ ] Trace the shape flow:
  ```
  x: (N, H) → gate_logits: (N, E) → gate_probs: softmax → top_indices, top_weights: (N, K)
  ```
- [ ] Understand: Renormalization of top-K weights to sum to 1

### Step 2: Understand Load Balancing Loss

**File:** `moe.py` - `load_balancing_loss()` function (lines 197-242)

- [ ] Read: The complete function
- [ ] Understand the formula:
  ```
  f_e = (# tokens routed to e) / (N * K)  # fraction of selections
  p_e = mean(gate_probs[:, e])            # mean routing probability
  L_aux = α * E * Σ_e (f_e * p_e)
  ```
- [ ] Why: Penalizes routing that concentrates on few experts

### Step 3: Understand Expert Processing

**File:** `moe.py` - `Expert` and `ExpertGroup` classes

- [ ] Read: `Expert.forward()` - Standard FFN (up → GELU → down)
- [ ] Read: `ExpertGroup.forward()` - Processes tokens by expert
- [ ] Key: Offsets array for slicing tokens per expert

```python
# ExpertGroup processes tokens sorted by expert ID
for e in range(self.num_local_experts):
    x_e = x_sorted[offsets[e]:offsets[e+1]]  # Tokens for expert e
    y_e = self.experts[e](x_e)                # Process
    y_sorted[offsets[e]:offsets[e+1]] = y_e
```

### Step 4: Trace Full MoE Forward

**File:** `moe.py` - `MoELayer.forward()` (lines 287-399)

This is the complete MoE flow. Study in stages:

**Stage 1: Flatten and Route (lines 310-326)**
```python
x_flat = rearrange(x, 'b s h -> (b s) h')           # Flatten
top_indices, top_weights, gate_probs = self.router(x_flat)  # Route
aux_loss = load_balancing_loss(...)                  # Compute loss
```

**Stage 2: Expand for K Selections (lines 328-334)**
```python
expanded_x = repeat(x_flat, 'n h -> (n k) h', k=self.top_k)  # N → N*K tokens
expanded_indices = rearrange(top_indices, 'n k -> (n k)')     # Flatten indices
```

**Stage 3: Sort by Expert (lines 336-340)**
```python
sort_idx = torch.argsort(expanded_indices, stable=True)
x_sorted = expanded_x[sort_idx]  # Group tokens by expert
```

**Stage 4: Process Experts (lines 381-385 for non-EP case)**
```python
local_counts = torch.bincount(local_indices, minlength=self.num_local_experts)
y_sorted = self.expert_group(x_sorted, local_counts, local_indices)
```

**Stage 5: Unsort and Reduce (lines 387-397)**
```python
inverse_sort = torch.argsort(sort_idx)
y_expanded = y_sorted[inverse_sort]                    # Restore order
y_weighted = y_expanded * weights_expanded.unsqueeze(-1)  # Apply weights
y = rearrange(y_weighted, '(n k) h -> n k h', k=self.top_k).sum(dim=1)  # Sum K experts
```

### Step 5: Run MoE Tests

- [ ] Run: MoE unit tests (no distributed required)
```bash
python tests/test_moe.py
```

- [ ] Run: Training with MoE enabled
```bash
python train.py --device auto --num_experts 8 --top_k 2 --moe_freq 2 --max_iters 100
```

---

## Concept Bridge: Token Routing

**Theory** (from course):
> Each token is routed to the top-K experts based on a learned gating function. The outputs are weighted and combined.

**Implementation** (in codebase):

```python
# moe.py Router.forward() - simplified
gate_logits = self.gate(x)                          # (N, E)
gate_probs = F.softmax(gate_logits, dim=-1)         # (N, E)
top_weights, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # (N, K)
top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)  # Renormalize
```

**Connection:**
- Course teaches the gating concept
- Code shows exact softmax → topk → renormalize flow
- Einops makes shape transformations explicit

---

## MoE Visualization

```
Input: x[B, S, H]
         │
         ▼ flatten
     x_flat[N, H]  where N = B*S
         │
         ▼ Router
  ┌──────┴──────┐
  │ top_indices │  (N, K) - which K experts per token
  │ top_weights │  (N, K) - contribution weights
  └──────┬──────┘
         │
         ▼ expand × K
   expanded_x[N*K, H]
         │
         ▼ sort by expert
    x_sorted[N*K, H]  - tokens grouped by expert
         │
    ┌────┼────┬────┬────┐
    │    │    │    │    │
   E0   E1   E2   E3  ...  (each expert processes its tokens)
    │    │    │    │    │
    └────┼────┴────┴────┘
         │
         ▼ unsort + weight + sum
      y[N, H]
         │
         ▼ reshape
  Output: y[B, S, H]
```

---

## Hands-On Exercises

### Exercise 1: Visualize Expert Load

Add logging to see expert utilization:

```python
# In MoELayer.forward(), after computing aux_loss
counts = torch.bincount(top_indices.flatten(), minlength=self.num_experts)
print(f"Expert counts: {counts.tolist()}")
print(f"Load std: {counts.float().std():.2f}")
```

### Exercise 2: Break Load Balancing

1. Set `aux_loss_coef = 0.0` in config
2. Train for 1000 steps
3. Observe expert collapse (few experts get all tokens)

### Exercise 3: Trace Permutation

Add prints to verify the sort/unsort cycle:

```python
print(f"Before sort: indices[:10] = {expanded_indices[:10]}")
print(f"After sort: indices_sorted[:10] = {indices_sorted[:10]}")
# ... after unsort
print(f"Verify identity: {torch.allclose(x_flat, y_expanded_check)}")
```

---

## Verification Checklist

- [ ] Can explain: "MoE routes tokens to subset of experts"
- [ ] Can trace: Token flow through expand → sort → process → unsort → reduce
- [ ] Understand: Why load balancing loss prevents expert collapse
- [ ] Successfully ran `test_moe.py` with all tests passing
- [ ] Can calculate: Activated parameters with 8 experts, top-2

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Expert collapse | aux_loss_coef too low | Increase to 0.01-0.1 |
| Unused experts | Random init imbalance | Train longer, use higher aux_loss |
| NCCL timeout (with DP) | Some experts have None grads | Zero-init grads for unused experts |
| OOM | Too many experts | Reduce num_experts or expert hidden size |

---

## Next Phase

MoE adds more parameters without proportional FLOP increase. But individual linear layers can still be large. Tensor Parallelism splits those layers across GPUs.

[Continue to Phase 5: Tensor Parallelism →](05-Phase-TP.md)
