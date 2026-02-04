# Phase 2: ZeRO & FSDP Concepts

**Week:** 2  
**Estimated Time:** 4-5 hours  
**Prerequisites:** Phase 1 (Data Parallelism)

---

## Learning Objectives

By completing this phase, you will:

1. **Understand** ZeRO Stages 1, 2, 3 and their memory savings
2. **Compare** ZeRO with vanilla DP (memory vs communication tradeoffs)
3. **Recognize** FSDP as PyTorch's ZeRO implementation
4. **Know** when to use each ZeRO stage

---

## ⚠️ Theory-Only Phase

**Note:** The heiretsu codebase implements **manual DP**, not ZeRO/FSDP. This phase is theory-focused using course content.

**Why?** The codebase prioritizes understanding fundamentals. ZeRO/FSDP abstracts away details that TP/PP make explicit.

---

## Theory (Course Content)

### Session: ZeRO Stage 1 & 2 (1 hour)

- [ ] Watch: **ZeRO: Stage 1 & 2** lecture
- [ ] Understand the memory breakdown of a training step:

| Component | Memory | ZeRO Stage |
|-----------|--------|------------|
| Parameters (W) | Φ | Stage 3 shards |
| Gradients (∇W) | Φ | Stage 2 shards |
| Optimizer states (m, v) | 2Φ - 8Φ | Stage 1 shards |
| Activations | Variable | Activation checkpointing |

- [ ] Key insight: Stage 1 shards **optimizer states** (biggest win for Adam)

### Session: ZeRO Stage 3 and Efficient Strategies (1.5 hours)

- [ ] Watch: **ZeRO Stage 3** lecture
- [ ] Understand: All-gather before forward, reduce-scatter after backward
- [ ] Key tradeoff: More communication, but fits larger models
- [ ] Compare with TP: ZeRO Stage 3 vs Tensor Parallelism

### Session: PyTorch FSDP (1 hour)

- [ ] Watch: **PyTorch FSDP** lecture
- [ ] Understand: FSDP = ZeRO Stage 3 in PyTorch native
- [ ] Key API:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

### Guest Lecture: Overview of ZeRO

- [ ] Watch: **Sylvain Gugger** (Jane Street) on ZeRO overview
- [ ] Get the historical context and DeepSpeed origins

---

## Concept Deep Dive: ZeRO Stages

### ZeRO Stage 1: Optimizer State Sharding

```
Before (DP):                    After (ZeRO-1):
┌─────────────┐                 ┌─────────────┐
│ GPU 0       │                 │ GPU 0       │
│ W (full)    │                 │ W (full)    │
│ ∇W (full)   │                 │ ∇W (full)   │
│ m,v (full)  │ ◄─ 8Φ memory    │ m,v (1/N)   │ ◄─ 8Φ/N memory
└─────────────┘                 └─────────────┘
```

**Communication:** All-reduce gradients + scatter optimizer updates

### ZeRO Stage 2: Gradient Sharding

```
┌─────────────┐
│ GPU 0       │
│ W (full)    │
│ ∇W (1/N)    │ ◄─ Only keeps gradients for params it optimizes
│ m,v (1/N)   │
└─────────────┘
```

**Communication:** Reduce-scatter gradients (instead of all-reduce)

### ZeRO Stage 3: Parameter Sharding

```
┌─────────────┐
│ GPU 0       │
│ W (1/N)     │ ◄─ All-gather before forward/backward
│ ∇W (1/N)    │
│ m,v (1/N)   │
└─────────────┘
```

**Communication:** All-gather params for compute, reduce-scatter grads

---

## Comparison: ZeRO vs TP vs PP

| Method | What's Sharded | Communication | Best For |
|--------|----------------|---------------|----------|
| **DP** | Nothing (replicated) | All-reduce grads | Small models |
| **ZeRO-1** | Optimizer states | All-reduce + scatter | Medium models |
| **ZeRO-2** | + Gradients | Reduce-scatter | Larger models |
| **ZeRO-3** | + Parameters | All-gather + reduce-scatter | Very large models |
| **TP** | Weight matrices | All-reduce activations | Large models, low latency |
| **PP** | Layers | P2P send/recv | Very deep models |

---

## Why Heiretsu Uses Manual DP

The codebase teaches parallelism **fundamentals**:

1. **DP** (`dp.py`): Shows exactly what `all_reduce` does
2. **TP** (`tp_linear.py`): Shows column/row sharding explicitly
3. **PP** (`pipeline.py`): Shows send/recv and microbatching

ZeRO/FSDP would abstract these away. For production, use FSDP. For learning, understand the primitives first.

---

## Hands-On Alternatives

Since we don't have ZeRO in codebase, try these:

### Alternative 1: Memory Profiling

Compare memory usage of DP vs theoretical ZeRO:

```python
# Add to train.py after model creation
import torch.cuda
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Alternative 2: External ZeRO Experiment

If you have DeepSpeed installed:

```python
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={"zero_optimization": {"stage": 2}}
)
```

### Alternative 3: PyTorch FSDP Quick Test

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap model with FSDP
model = FSDP(model)
# Training proceeds normally
```

---

## Verification Checklist

- [ ] Can explain: What does each ZeRO stage shard?
- [ ] Can calculate: Memory savings for 1B param model with ZeRO-1 on 8 GPUs
- [ ] Understand: ZeRO-3 vs TP - when to use which?
- [ ] Know: FSDP = PyTorch's ZeRO-3 implementation

---

## Knowledge Check

**Q1:** You have a 7B model and 8 A100-80GB GPUs. Which ZeRO stage do you need?

<details>
<summary>Answer</summary>

**ZeRO Stage 2 or 3.** 7B params × 4 bytes = 28GB for weights alone. With Adam optimizer states (2× params), that's 84GB total per GPU in vanilla DP. ZeRO-2 shards optimizer states and gradients, reducing to ~35GB/GPU. ZeRO-3 shards everything, reducing to ~14GB/GPU.

</details>

**Q2:** Why might TP be preferred over ZeRO-3 for inference?

<details>
<summary>Answer</summary>

TP has lower latency for inference because:
1. All-gather in ZeRO-3 happens every forward pass
2. TP's all-reduce is on smaller activation tensors
3. TP computation and communication can be overlapped

</details>

---

## Next Phase

ZeRO optimizes memory for DP-style training. But what if your model is too deep for one GPU even with full sharding? That's where Pipeline Parallelism comes in.

[Continue to Phase 3: Pipeline Parallelism →](03-Phase-PP.md)
