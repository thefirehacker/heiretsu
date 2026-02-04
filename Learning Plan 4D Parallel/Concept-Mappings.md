# Concept-to-Code Mappings

This document bridges course theory with codebase implementations.

---

## Quick Reference Table

| Course Concept | Week | Code File | Key Function/Class | Line Range |
|----------------|------|-----------|-------------------|------------|
| Gradient Averaging | 1 | `dp.py` | `average_gradients()` | 13-31 |
| Batch Sharding | 1 | `train.py` | `RandomBatchLoader` | - |
| Process Groups | 1 | `topo.py` | `build_groups()` | 48-93 |
| Layer Partitioning | 3 | `pipeline.py` | `_partition_layers()` | 19-33 |
| Microbatch Schedule | 3 | `pipeline.py` | `forward_backward()` | 199-294 |
| P2P Communication | 3 | `pipeline.py` | `_p2p()` | 178-191 |
| Top-K Routing | 3 | `moe.py` | `Router.forward()` | 47-77 |
| Load Balancing Loss | 3 | `moe.py` | `load_balancing_loss()` | 197-242 |
| Expert Processing | 3 | `moe.py` | `ExpertGroup.forward()` | 149-194 |
| Column-Parallel | 4 | `tp_linear.py` | `ColumnParallelLinear` | 87-167 |
| Row-Parallel | 4 | `tp_linear.py` | `RowParallelLinear` | 245-337 |
| TP Initialization | 4 | `tp_linear.py` | `reset_parameters()` | 123-145 |
| All-to-All Dispatch | Bonus | `ep_comm.py` | `all_to_all_dispatch()` | - |
| All-to-All Combine | Bonus | `ep_comm.py` | `all_to_all_combine()` | - |
| 4D Rank Layout | Bonus | `topo.py` | `rank_from_coords()` | 44-45 |

---

## Detailed Concept Bridges

### Bridge 1: Data Parallelism - Gradient Averaging

**Theory** (Week 1 - DDP from Scratch):
> Each GPU processes different mini-batches. After backward pass, gradients are averaged across all GPUs to keep models synchronized.

**Formula:**
```
∇W_avg = (1/N) × Σᵢ ∇Wᵢ
```

**Implementation** (`dp.py`):
```python
def average_gradients(model, dp_group):
    world = dist.get_world_size(group=dp_group)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)  # MoE: unused experts
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
        p.grad.div_(world)  # Average = SUM / world_size
```

**Key Insight:** The code handles MoE's unused experts by initializing zero gradients, preventing NCCL hangs.

---

### Bridge 2: Pipeline Parallelism - Microbatching

**Theory** (Week 3 - Pipeline Parallelism):
> Microbatching fills the pipeline to reduce bubble overhead. With M microbatches and P stages, bubble fraction = (P-1)/M.

**Formula:**
```
Bubble Fraction = (P - 1) / M
Efficiency = M / (M + P - 1)
```

**Implementation** (`pipeline.py`):
```python
# Forward phase: fill pipeline
for m in range(micro_batches):
    # Receive from previous stage (except first)
    if not self.is_first:
        self._p2p(None, recv_x, self.prev_rank, self._fwd_tag(m))
    
    # Forward through local blocks
    x, aux_loss = self.stage.forward_blocks(x)
    
    # Send to next stage (except last)
    if not self.is_last:
        self._p2p(boundary.detach(), None, self.next_rank, self._fwd_tag(m))

# Backward phase: drain pipeline (reverse order)
for m in reversed(range(micro_batches)):
    # ... gradient flow in reverse
```

**Key Insight:** Tags (`_fwd_tag(m)`, `_bwd_tag(m)`) prevent microbatch message mixup.

---

### Bridge 3: Tensor Parallelism - Column/Row Sharding

**Theory** (Week 4 - Tensor Parallelism):
> Column-parallel shards output features (rows of weight matrix). Row-parallel shards input features (columns). Pairing them avoids redundant communication.

**Sharding Pattern:**
```
Column-Parallel:  W[out, in] → W_shard[out/TP, in]
Row-Parallel:     W[out, in] → W_shard[out, in/TP]
```

**Implementation** (`tp_linear.py`):
```python
class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        x = copy_to_model_parallel_region(x, self.tp_group)  # Identity fwd, all-reduce bwd
        y_shard = einops.einsum(x2d, W_shard, 'bs d, op d -> bs op')
        return y_shard  # (B, S, out/TP)

class RowParallelLinear(nn.Module):
    def forward(self, x):
        y_partial = einops.einsum(x_shard, W_shard, 'bs d, op d -> bs op')
        y_full = reduce_from_model_parallel_region(y_partial, self.tp_group)  # All-reduce
        return y_full  # (B, S, out)
```

**Key Insight:** Communication is asymmetric - Column-parallel all-reduces in backward, Row-parallel in forward.

---

### Bridge 4: MoE - Token Routing

**Theory** (Week 3 - MoE Training):
> Each token is routed to top-K experts via a learned gating function. Outputs are weighted and combined.

**Formula:**
```
gate_logits = x @ W_gate           # (N, E)
gate_probs = softmax(gate_logits)  # (N, E)
top_k_weights, top_k_indices = topk(gate_probs, k=K)
output = Σₖ (weight_k × Expert_k(x))
```

**Implementation** (`moe.py`):
```python
class Router(nn.Module):
    def forward(self, x):
        gate_logits = self.gate(x)                    # (N, E)
        gate_probs = F.softmax(gate_logits, dim=-1)   # (N, E)
        top_weights, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)  # Renormalize
        return top_indices, top_weights, gate_probs
```

**Key Insight:** Renormalization ensures weights sum to 1 for proper weighted combination.

---

### Bridge 5: Load Balancing Loss

**Theory** (Week 3 - MoE Training):
> Without balancing, some experts may receive all tokens while others get none. Auxiliary loss penalizes imbalanced routing.

**Formula:**
```
f_e = (# tokens routed to e) / (N × K)  # Fraction of selections
p_e = mean(gate_probs[:, e])            # Mean routing probability
L_aux = α × E × Σ_e (f_e × p_e)
```

**Implementation** (`moe.py`):
```python
def load_balancing_loss(gate_probs, top_indices, num_experts, top_k, aux_loss_coef):
    # f_e: fraction of tokens routed to each expert
    counts = torch.zeros(num_experts, device=gate_probs.device)
    counts.scatter_add_(0, top_indices.flatten(), torch.ones_like(top_indices.flatten()))
    f = counts / (N * top_k)
    
    # p_e: mean routing probability per expert
    p = gate_probs.mean(dim=0)
    
    # L_aux = α × E × Σ(f_e × p_e)
    aux_loss = aux_loss_coef * num_experts * (f * p).sum()
    return aux_loss
```

**Key Insight:** The product `f_e × p_e` penalizes experts that are both selected often AND have high probability.

---

### Bridge 6: Expert Parallelism - All-to-All

**Theory** (Bonus - Expert Parallelism):
> With many experts, each GPU owns a subset. All-to-All sends tokens to the GPU owning the target expert.

**Communication Pattern:**
```
Dispatch: Each GPU sends tokens to GPU owning target expert
          Variable-size send (depends on routing)
Combine:  Reverse dispatch - tokens return to origin GPU
```

**Implementation** (`ep_comm.py` + `moe.py`):
```python
# Determine destination EP rank for each token
ep_indices = indices_sorted // experts_per_rank  # Which EP rank owns each expert
send_counts = torch.bincount(ep_indices, minlength=ep_world_size)

# Dispatch tokens to EP ranks owning target experts
x_recv, recv_counts, idx_recv = all_to_all_dispatch(x_sorted, send_counts, ep_group, ...)

# Process locally-owned experts
local_indices = idx_recv - (ep_rank * num_local_experts)
y_local = expert_group(x_recv, local_counts, local_indices)

# Return processed tokens to origin
y_return = all_to_all_combine(y_local, send_counts, recv_counts, ep_group, ...)
```

**Key Insight:** `send_counts` and `recv_counts` are swapped between dispatch and combine.

---

### Bridge 7: 4D Topology

**Theory** (Bonus - Multi-dimensional Parallelism):
> Combining DP, TP, PP, EP requires careful rank organization. Each GPU belongs to one group per dimension.

**Rank Layout Formula:**
```
global_rank = dp*(EP*PP*TP) + ep*(PP*TP) + pp*TP + tp
```

**Implementation** (`topo.py`):
```python
def rank_from_coords(dp_rank, ep_rank, pp_rank, tp_rank, dp, ep, tp, pp):
    return dp_rank * (ep * pp * tp) + ep_rank * (pp * tp) + pp_rank * tp + tp_rank

def coords_from_rank(rank, dp, ep, tp, pp):
    tp_rank = rank % tp
    pp_rank = (rank // tp) % pp
    ep_rank = (rank // (tp * pp)) % ep
    dp_rank = rank // (tp * pp * ep)
    return dp_rank, ep_rank, pp_rank, tp_rank
```

**Key Insight:** TP is innermost (contiguous ranks share TP group) for fastest NVLink communication.

---

### Bridge 8: TP Initialization (SmolLM3 Bug)

**Theory** (Week 4 + SmolLM3 Playbook):
> Using identical seeds across TP ranks causes correlated weight initialization, hurting convergence.

**Bug:**
```python
# BAD: All TP ranks get same weights
torch.manual_seed(seed)
nn.init.normal_(self.weight)
```

**Fix** (`tp_linear.py`):
```python
def reset_parameters(self, std=0.02, base_seed=1337):
    # Unique seed per TP-rank, layer, and module
    seed = base_seed + self.tp_rank * 1000000 + self.layer_idx * 100 + self.module_id
    
    gen = torch.Generator(device=self.weight.device)
    gen.manual_seed(seed)
    
    self.weight.normal_(mean=0.0, std=std, generator=gen)
```

**Key Insight:** Per-rank seeds ensure diversity in sharded weights while maintaining determinism.

---

## Einops Shape Annotations Reference

The codebase uses consistent shape annotations:

| Notation | Meaning | Example |
|----------|---------|---------|
| `b` | Batch size | 8 |
| `s` | Sequence length | 1024 |
| `h` | Hidden dimension | 768 |
| `n` | Flattened tokens (b*s) | 8192 |
| `k` | Top-K experts | 2 |
| `e` | Number of experts | 8 |
| `tp` | TP world size | 2 |
| `nh` | Number of heads | 12 |
| `dh` | Head dimension | 64 |

**Common patterns:**
```python
# Flatten batch and sequence
x_flat = rearrange(x, 'b s h -> (b s) h')  # (N, H)

# Split for TP sharding
x_shards = rearrange(x, '... (tp d) -> tp ... d', tp=2)

# Expand for K selections
expanded = repeat(x, 'n h -> (n k) h', k=2)  # (N*K, H)

# Reshape for multi-head attention
q = rearrange(q, 'b s (nh dh) -> b nh s dh', nh=12)
```

---

## Communication Primitive Reference

| Collective | Forward | Backward | Use Case |
|------------|---------|----------|----------|
| `all_reduce` | Sum all → all | Identity | DP gradients, TP row-parallel |
| `reduce_scatter` | Sum → sharded | All-gather | ZeRO-2 gradients |
| `all_gather` | Sharded → all | Reduce-scatter | ZeRO-3 params, PP boundary |
| `all_to_all` | Variable exchange | Inverse all-to-all | EP token dispatch |
| `send/recv` | P2P transfer | P2P transfer | PP activations |

---

## Test Command Reference

| Concept | Test Command | Expected Output |
|---------|--------------|-----------------|
| DP | `torchrun --nproc=2 tests/forward_dp.py` | `diff_max 0.0` |
| TP | `torchrun --nproc=2 tests/forward_tp.py` | `diff_max < 1e-6` |
| PP | `torchrun --nproc=2 tests/forward_pp.py --pp 2` | `diff ≈ 0` |
| MoE | `python tests/test_moe.py` | All tests pass |
| Combined | `bash tests/run_full_suite.sh` | All configs PASS |
