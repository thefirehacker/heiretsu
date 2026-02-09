# Phase 7: 4D Parallelism Integration

**Week:** Bonus  
**Estimated Time:** 6-8 hours  
**Prerequisites:** All previous phases (DP, ZeRO concepts, PP, MoE, TP, EP)

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** how DP, TP, PP, EP work together
2. **Configure** multi-dimensional parallelism for different model sizes
3. **Debug** complex distributed training issues
4. **Deploy** training to cloud (Modal)

---

## Theory (Course Content)

### Guest Lectures

- [ ] Watch: **Multi-dimensional parallelism** - Ferdinand Mom (Hugging Face)
- [ ] Watch: **2D Parallelism with Axolotl** - Wing Lian
- [ ] Watch: **Back-propagation from Scratch** - Daniel Han (UnslothAI)
  - Understand gradient flow to debug distributed issues

### Additional Resources

- [ ] Review: **DiLoCo (Part 1 & 2)** - Sami Jaghouar
  - Decentralized training at global scale

---

## Practice (Codebase)

### Step 1: Master the Topology

**File:** `topo.py` - The orchestrator of 4D parallelism

The rank layout formula is crucial:
```python
global_rank = dp*(EP*PP*TP) + ep*(PP*TP) + pp*TP + tp
```

- [ ] Trace: How 8 GPUs are organized with dp=2, ep=1, tp=2, pp=2
- [ ] Draw: The 3D grid of (dp, tp, pp) coordinates

```
Example: 8 GPUs with DP=2, TP=2, PP=2

Rank | dp | tp | pp | Role
-----|----|----|----|--------------------------
  0  |  0 |  0 |  0 | DP replica 0, TP shard 0, PP stage 0 (first)
  1  |  0 |  1 |  0 | DP replica 0, TP shard 1, PP stage 0 (first)
  2  |  0 |  0 |  1 | DP replica 0, TP shard 0, PP stage 1 (last)
  3  |  0 |  1 |  1 | DP replica 0, TP shard 1, PP stage 1 (last)
  4  |  1 |  0 |  0 | DP replica 1, TP shard 0, PP stage 0 (first)
  5  |  1 |  1 |  0 | DP replica 1, TP shard 1, PP stage 0 (first)
  6  |  1 |  0 |  1 | DP replica 1, TP shard 0, PP stage 1 (last)
  7  |  1 |  1 |  1 | DP replica 1, TP shard 1, PP stage 1 (last)
```

### Step 2: Understand Process Group Relationships

Each parallelism dimension has its own process group:

| Group | Same Coordinates | Varying | Communication |
|-------|------------------|---------|---------------|
| `tp_group` | dp, ep, pp | tp | All-reduce activations |
| `pp_group` | dp, ep, tp | pp | P2P send/recv |
| `ep_group` | dp, pp, tp | ep | All-to-All tokens |
| `dp_group` | ep, pp, tp | dp | All-reduce gradients |

- [ ] Find: Where each group is created in `build_groups()`
- [ ] Verify: Each rank belongs to exactly one group per dimension

### Step 3: Trace Full Training Step

**File:** `train.py` - Follow a complete training iteration

```python
# 1. Data Loading (DP-aware)
idx, targets = loader.next_batch(device)  # Each DP rank gets different data

# 2. Forward Pass
if args.pp == 1:
    # Non-PP: Direct forward through full model
    logits, loss, aux_loss = model(idx, targets)  # TP happens inside
else:
    # PP: Microbatched forward-backward
    loss, aux = engine.forward_backward(...)

# 3. Gradient Sync (DP)
average_gradients(model, topo.dp_group)  # All-reduce across DP ranks

# 4. Optimizer Step
optimizer.step()  # All ranks update (should have identical grads now)
```

### Step 4: Run Full 3D Test Suite

- [ ] Run: Complete test suite (4+ GPUs required)
```bash
bash tests/run_full_suite.sh
```

Expected configurations tested:
- Single dimension: `dp=4`, `tp=4`, `pp=4`
- 2D combinations: `dp=2 tp=2`, `dp=2 pp=2`, `tp=2 pp=2`
- Full 3D (8 GPUs): `dp=2 tp=2 pp=2`

### Step 5: Run 4D with MoE (8 GPUs)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --dp 2 --ep 2 --tp 2 --pp 1 \
    --num_experts 8 --top_k 2 --moe_freq 2 \
    --n_layer 12 --n_head 12 --n_embed 768 \
    --max_iters 100
```

### Step 6: Deploy to Cloud (Modal)

**File:** `modal_train.py` - Production deployment

- [ ] Read: How Modal app is configured
- [ ] Understand: Volume mounts for data and traces
- [ ] Find: Training config (4x A100, DP=2, TP=2, MoE)

```bash
# Deploy training to Modal
modal run modal_train.py
```

---

## Configuration Guide

### Choosing Parallelism Dimensions

| Model Size | Recommended Config | Why |
|------------|-------------------|-----|
| < 1B | DP only | Model fits on one GPU |
| 1-10B | DP + TP | Shard large layers |
| 10-100B | DP + TP + PP | Shard layers and split stages |
| 100B+ MoE | DP + TP + PP + EP | Full 4D for massive MoE |

### Memory Estimation

```
Per-GPU Memory ≈ 
  (Parameters / TP / PP) × (1 + optimizer_mult)
  + Activations × (batch / DP) × (seq / CP if using)
  + Gradients / TP / PP
```

For Adam optimizer: `optimizer_mult ≈ 2` (m and v states)

### Throughput Estimation

```
Tokens/second = (batch_size × seq_len × DP × world_size) / step_time

Effective batch = batch_per_gpu × grad_accum × DP
```

---

## Debugging Multi-Dimensional Issues

### Common Patterns

**Issue: NCCL Timeout**
```
WorkNCCL(...) Watchdog caught collective operation timeout
```

**Diagnosis:**
1. Which collective? (DP all-reduce, TP all-reduce, PP P2P, EP All-to-All)
2. Which ranks involved? (Check group membership)
3. Is one rank not participating? (grad=None, missing data)

**Fix:** Usually grad=None for unused MoE experts → initialize zero grads

---

**Issue: Numerical Divergence**
```
Loss explodes after N steps
```

**Diagnosis:**
1. Check gradient norms across DP ranks (should be identical)
2. Check weight norms across TP ranks (should differ for sharded layers)
3. Verify PP boundary activations match

**Fix:** Usually initialization issue → check per-rank seeds for TP

---

**Issue: OOM on Some Ranks**
```
CUDA out of memory on rank X
```

**Diagnosis:**
1. Is PP balanced? (Last stage has LM head overhead)
2. Is MoE load balanced? (Some experts get more tokens)
3. Is batch size even across DP?

**Fix:** Adjust layer partition, reduce microbatch size, add capacity factor for MoE

---

## Hands-On Exercises

### Exercise 1: Visualize Rank Topology

Add debug output at startup:

```python
# In train.py after init_topology
print(f"Rank {topo.rank}: dp={topo.dp_rank}, ep={topo.ep_rank}, "
      f"tp={topo.tp_rank}, pp={topo.pp_rank}")
print(f"  TP group ranks: {list(dist.get_process_group_ranks(topo.tp_group))}")
print(f"  DP group ranks: {list(dist.get_process_group_ranks(topo.dp_group))}")
```

### Exercise 2: Profile Communication

Add timing for each collective type:

```python
# Wrap each collective with timing
def timed_all_reduce(tensor, group, name):
    torch.cuda.synchronize()
    start = time.time()
    dist.all_reduce(tensor, group=group)
    torch.cuda.synchronize()
    print(f"{name} all-reduce: {(time.time()-start)*1000:.2f}ms")
```

### Exercise 3: Compare Configurations

Train the same model with different configs and compare:

```bash
# Config A: DP only
torchrun --nproc=4 train.py --dp 4 --max_iters 100

# Config B: DP + TP
torchrun --nproc=4 train.py --dp 2 --tp 2 --max_iters 100

# Config C: TP + PP
torchrun --nproc=4 train.py --tp 2 --pp 2 --grad_accum 4 --max_iters 100
```

Compare: throughput, memory usage, loss curve

---

## Verification Checklist

- [ ] Can explain: "4D parallelism = DP + TP + PP + EP working together"
- [ ] Can configure: Parallelism for a given model size and GPU count
- [ ] Can debug: NCCL timeouts by identifying which collective failed
- [ ] Successfully ran: Full test suite with all configs passing
- [ ] Successfully deployed: Training to Modal cloud

---

## Final Project Ideas

### Project 1: Benchmark Different Configs

- Create a benchmark script comparing throughput across configs
- Plot: tokens/sec vs GPU count for DP, TP, PP, and combinations
- Analyze: Communication overhead at each scale

### Project 2: Train a Real MoE

- Use Modal to train a 355M MoE model
- Track: Loss curves, expert utilization, load balance
- Compare: Dense vs MoE at same FLOPs

### Project 3: Implement Missing Features

- Add activation checkpointing for PP stages
- Implement 1F1B schedule (instead of GPipe)
- Add sequence parallelism for long contexts

---

## Graduation Checklist

After completing all phases, verify you can:

- [ ] **Explain** each parallelism type in one sentence
- [ ] **Implement** gradient averaging (DP)
- [ ] **Trace** column/row parallel layers (TP)
- [ ] **Debug** pipeline deadlocks (PP)
- [ ] **Configure** MoE with load balancing (MoE)
- [ ] **Understand** All-to-All dispatch (EP)
- [ ] **Deploy** multi-GPU training to cloud
- [ ] **Diagnose** common distributed training failures

---

## Resources for Continued Learning

| Resource | Link | Focus |
|----------|------|-------|
| TorchTitan | github.com/pytorch/torchtitan | Production 3D/4D parallelism |
| Megatron-LM | github.com/NVIDIA/Megatron-LM | NVIDIA's parallelism library |
| DeepSpeed | github.com/microsoft/DeepSpeed | ZeRO and large-scale training |
| SmolLM3 Playbook | huggingface.co/blog/smollm3 | Real-world training lessons |

---

Congratulations on completing the 4D Parallelism Learning Plan!

[Return to Overview →](00-Overview.md)
