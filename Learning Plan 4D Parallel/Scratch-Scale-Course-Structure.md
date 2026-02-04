# Scratch to Scale: Large-Scale Training in the Modern World

**Course URL:** https://maven.com/walk-with-code/scratch-to-scale  
**Instructor:** Zachary Mueller (HuggingFace Accelerate Technical Lead)  
**Duration:** 4 Weeks  
**Compute Credits:** $1000 Modal + $1000 Lambda

---

## 4D Parallelism Coverage Mapping

| Parallelism | Week | Sessions |
|-------------|------|----------|
| **Data Parallelism (DP)** | Week 1 | DDP from Scratch, DataLoader Workshop |
| **ZeRO (Sharded DP)** | Week 2 | ZeRO Stage 1, 2, 3 + PyTorch FSDP |
| **Pipeline Parallelism (PP)** | Week 3 | Pipeline Parallelism, PP with torchtitan |
| **Tensor Parallelism (TP)** | Week 4 | Tensor Parallelism, TP with TorchTitan |
| **Expert Parallelism (EP)** | Week 3 + Bonus | How to train your small MoE, Expert Parallelism guest lecture |

---

## Week 1: Data Parallelism & DataLoading

### Core Sessions

| Session | Duration | Topics |
|---------|----------|--------|
| **Distributed Data Parallelism From Scratch** | 1 hour | DDP implementation, all-reduce, gradient averaging |
| **DataLoader Workshop** | 1 hour | DistributedSampler, avoiding data bottlenecks |

### Key Concepts
- Batch sharding: `B_global = B_local * world_size`
- Gradient synchronization via all-reduce
- Distributed sampling strategies
- Data bottleneck prevention

---

## Week 2: ZeRO & FSDP

### Core Sessions

| Session | Duration | Topics |
|---------|----------|--------|
| **ZeRO: Stage 1 & 2** | 1 hour | Optimizer and gradient sharding |
| **ZeRO: Stage 3 and Efficient ZeRO Strategies** | 1.5 hours | Parameter sharding, efficiency tradeoffs |
| **PyTorch FSDP** | 1 hour | Native PyTorch ZeRO-3 implementation |

### Key Concepts
- ZeRO Stage 1: Optimizer state sharding
- ZeRO Stage 2: Gradient sharding
- ZeRO Stage 3: Parameter sharding
- Memory vs communication tradeoffs
- FSDP sharding strategies

---

## Week 3: Pipeline Parallelism & MoE

### Core Sessions

| Session | Duration | Topics |
|---------|----------|--------|
| **Pipeline Parallelism** | 1.5 hours | GPipe, microbatching, bubble overhead |
| **Hands-on FP8 Workshop** | 1 hour | FP8 on consumer + datacenter hardware |
| **Pipeline Parallelism with torchtitan** | 1 hour | Production PP implementation |
| **How to train your small MoE** | 1 hour | Routing, load balancing, training tips |

### Key Concepts
- Layer partitioning across stages
- Microbatch scheduling (GPipe, 1F1B)
- Bubble overhead formula: (P-1)/M
- P2P send/recv communication
- Top-K expert routing
- Load balancing auxiliary loss
- FP8 precision training

---

## Week 4: Tensor Parallelism

### Core Sessions

| Session | Duration | Topics |
|---------|----------|--------|
| **Tensor Parallelism** | 1.5 hours | Column/row parallel, communication patterns |
| **TensorParallelism with TorchTitan** | 1 hour | DTensor, production TP |

### Key Concepts
- Column-parallel (output sharding)
- Row-parallel (input sharding)
- Communication: all-reduce activations
- TP initialization (per-rank seeds - SmolLM3 fix)
- DTensor abstraction

---

## Bonus: Guest Speakers (All 10+ Included)

### Essential for 4D Parallelism

| Topic | Speaker | Company | Priority |
|-------|---------|---------|----------|
| **Expert Parallelism** | Matej Sirovatka | Hugging Face | **Essential** for EP |
| **Async Tensor Parallelism** | Less Wright | Meta | Advanced TP techniques |
| **Introduction to TorchTitan** | Wanchao Liang | TorchTitan | DTensor + large-scale pretraining |
| **Multi-dimensional Parallelism** | Ferdinand Mom | Hugging Face | **4D combined** |
| **2D Parallelism with Axolotl** | Wing Lian | Axolotl | Practical 2D implementation |
| **Back-propagation from Scratch** | Daniel Han | UnslothAI | Gradient math (hand-written) |

### Advanced Topics

| Topic | Speaker | Company |
|-------|---------|---------|
| **Arctic Long Sequence Training** | Tunji Ruwase | Snowflake |
| **Hyper-optimizing LLMs with MoE, MLA** | Elie Bakouch | Hugging Face |
| **DiLoCo (Part 1)** | Sami Jaghouar | Prime Intellect |
| **DiLoCo (Part 2)** | Sami Jaghouar | Prime Intellect |
| **A practitioner's guide to FP8** | Phuc Nguyen | Hugging Face |
| **Overview of ZeRO** | Sylvain Gugger | Jane Street |
| **MLX** | Prince Canuma | - |
| **Inference Deep Dive** | Marc Sun | Hugging Face |
| **State of Research in AI** | - | - |

### Fireside Chats
| Speaker | Company |
|---------|---------|
| **Yuxiang Wei** | Meta FAIR |

### Conference Talks - Applied Track

| Speaker | Company | Topic |
|---------|---------|-------|
| Robert Nishihara | Ray, Anyscale | Scaling across thousands of GPUs with Ray |
| Sami Jaghouar | Prime Intellect | Decentralized global-scale training |
| Tunji Ruwase | Snowflake | Efficient long-context training with Arctic |
| Prince Canuma | - | Local ML workloads using Apple Silicon + MLX |

### Conference Talks - Pretraining Track

| Speaker | Company | Topic |
|---------|---------|-------|
| Phuc Nguyen | Hugging Face | A practitioner's guide to FP8 |
| Elie Bakouch | Hugging Face | Hyper-optimizing LLMs with MoE, MLA & more |
| Daniel Han | UnslothAI | Speeding up training with Triton & custom kernels |

---

## Core Workshops Summary

| Workshop | Focus |
|----------|-------|
| **DDP from scratch** | Avoiding data bottlenecks |
| **ZeRO (Part 1)** | How model sharding enables scale |
| **ZeRO (Part 2)** | Efficiency tradeoffs and stage comparison |
| **Pipeline & Tensor Parallelism** | Solving communication slowdowns |
| **Multi-Dimensional Parallelism** | Combining all methods for throughput |

---

## Additional Workshops

| Workshop | Focus |
|----------|-------|
| **DataLoaders + Distributed** | Proper data loading at scale |
| **FP8 in the real world** | Consumer + datacenter hardware |
| **PyTorch traces** | Verify implementations |

---

## Daniel Han's Back-propagation Session

Hand-written derivations of modern transformer (LLaMA-style):

```
residual = X
X = X / sqrt(X² + ε) * ω        # RMSNorm
Q, K, V = XW_Q, XW_K, XW_V      # Projections
Q, K = RoPE(Q, K)               # Rotary Position Embeddings
A = σ((QK^T / √d) + M) V        # Scaled Dot-Product Attention with mask
O = A W_O                        # Output projection
X = residual + O                 # Residual connection

residual = X
X = X / sqrt(X² + ε) * ω        # RMSNorm (again for MLP)
G, U = X W_gate, X W_up         # SwiGLU gate and up projections
D = [f(G) * U] W_down           # SwiGLU activation and down projection
X = residual + D                 # Residual connection
```

**Covers:**
- RMSNorm (not LayerNorm)
- RoPE positional embeddings
- SwiGLU activation in MLP
- Complete gradient derivations by hand

---

## Resources

- **Distributed Operations Cheatsheet** - PDF with visuals of torch.distributed operations
- **Distributed Training Lexicon** - 49 terms with definitions
- **Course notebooks & code** - Detailed implementation notebooks
- **Class Discord** - Lifetime access

---

## Learning Plan Files

| File | Purpose |
|------|---------|
| [00-Overview.md](00-Overview.md) | Goals, prerequisites, structure |
| [01-Phase-DP.md](01-Phase-DP.md) | Data Parallelism |
| [02-Phase-ZeRO.md](02-Phase-ZeRO.md) | ZeRO Concepts (theory) |
| [03-Phase-PP.md](03-Phase-PP.md) | Pipeline Parallelism |
| [04-Phase-MoE.md](04-Phase-MoE.md) | Mixture of Experts |
| [05-Phase-TP.md](05-Phase-TP.md) | Tensor Parallelism |
| [06-Phase-EP.md](06-Phase-EP.md) | Expert Parallelism |
| [07-Phase-Integration.md](07-Phase-Integration.md) | 4D Integration |
| [Concept-Mappings.md](Concept-Mappings.md) | Theory-to-code bridges |
| [Progress-Tracker.md](Progress-Tracker.md) | Checklist for tracking |
