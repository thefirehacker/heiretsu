# moe-4d-parallel-minimal

**Minimal 4D parallelism (DP + TP + PP + EP) with MoE in pure PyTorch**

[![GitHub](https://img.shields.io/badge/GitHub-thefirehacker%2Fmoe--4d--parallel--minimal-blue)](https://github.com/thefirehacker/moe-4d-parallel-minimal)

> **Fork Notice:** This project is a fork of [heiretsu](https://github.com/ChinmayK0607/heiretsu) by [Chinmay Kak](https://github.com/ChinmayK0607).

---

## Why This Project?

Most educational distributed training repos stop at basic parallelism. This project fills the gap:

| Feature | [nanochat](https://github.com/karpathy/nanochat) | [Picotron](https://github.com/huggingface/picotron) | **moe-4d-parallel-minimal** |
|---------|---------|----------|------------------------|
| Data Parallelism (DP) | Yes (DDP) | Yes | Yes (manual) |
| Tensor Parallelism (TP) | No | Yes | Yes |
| Pipeline Parallelism (PP) | No | Yes | Yes |
| Context Parallelism (CP) | No | Yes | No |
| **Expert Parallelism (EP)** | No | No | **Yes** |
| **MoE Support** | No | No | **Yes** |
| Model Type | GPT | LLaMA | GPT + MoE |
| Purpose | Simple training | Educational | Educational + MoE |

**Key Differentiator:** The only minimal/educational repo with **Expert Parallelism** and **Mixture of Experts** support.

---

## What's Included

- **Data Parallelism (DP)**: Manual gradient averaging with all-reduce (no DDP/FSDP)
- **Tensor Parallelism (TP)**: Megatron-style column/row parallel linears for attention + MLP
- **Pipeline Parallelism (PP)**: GPipe-style microbatching with fill/drain schedule
- **Expert Parallelism (EP)**: All-to-All token dispatch across expert-owning GPUs
- **Mixture of Experts (MoE)**: Top-K routing, load balancing loss, configurable expert frequency
- **Mixed Precision**: Optional AMP (fp16/bf16)

### Architecture

```
moe-4d-parallel-minimal/
├── train.py          # Main training loop (integrates all parallelism)
├── gpt_model.py      # GPT model with TP/MoE support
├── topo.py           # Topology: process group management (DP/EP/TP/PP)
├── dp.py             # Data Parallelism: gradient averaging
├── tp_linear.py      # Tensor Parallelism: Column/Row parallel layers
├── pipeline.py       # Pipeline Parallelism: GPipe microbatching
├── moe.py            # MoE: Router, Experts, load balancing
├── ep_comm.py        # Expert Parallelism: All-to-All communication
└── tests/            # Verification tests for each parallelism type
```

---

## Quick Start

### Single GPU
```bash
python train.py --device auto
```

### Data Parallel (2 GPUs)
```bash
torchrun --standalone --nproc_per_node=2 train.py --dp 2
```

### Tensor Parallel (2 GPUs)
```bash
torchrun --standalone --nproc_per_node=2 train.py --tp 2
```

### Pipeline Parallel (2 GPUs)
```bash
torchrun --standalone --nproc_per_node=2 train.py --pp 2 --grad_accum_steps 4
```

### MoE with Expert Parallel (4 GPUs)
```bash
torchrun --standalone --nproc_per_node=4 train.py --dp 2 --ep 2 --num_experts 8 --top_k 2
```

### Full 4D Parallelism (8 GPUs)
```bash
torchrun --standalone --nproc_per_node=8 train.py --dp 2 --tp 2 --pp 2 --num_experts 8
```

---

## Test Suite

### Full Test Suite (4+ GPUs)

Runs forward parity, backward parity, gradient parity, and training smoke tests:

```bash
bash tests/run_full_suite.sh
```

**Configurations tested (4 GPUs):**
- `dp=4 tp=1 pp=1` (DP only)
- `dp=1 tp=4 pp=1` (TP only)
- `dp=1 tp=1 pp=4` (PP only)
- `dp=2 tp=2 pp=1` (DP + TP)
- `dp=2 tp=1 pp=2` (DP + PP)
- `dp=1 tp=2 pp=2` (TP + PP)

**Additional configs (8 GPUs):**
- `dp=2 tp=2 pp=2` (Full 3D)
- `dp=2 ep=2 tp=2 pp=1` (4D with MoE)

### Individual Tests

```bash
# TP parity test (2 GPUs)
torchrun --standalone --nproc_per_node=2 tests/tests_equiv.py --tp 2

# Parallel sanity (any config)
torchrun --standalone --nproc_per_node=4 tests/parallel_sanity.py --dp 2 --tp 2 --pp 1

# Gradient parity vs single-GPU baseline
torchrun --standalone --nproc_per_node=4 tests/grad_parity.py --dp 1 --tp 2 --pp 2

# MoE unit tests (no GPU required)
python tests/test_moe.py
```

### Expected Output

```
=== Summary ===
PASS (6):
  dp=4 tp=1 pp=1 (diff=0.0)
  dp=1 tp=4 pp=1 (diff=1.27e-07)
  dp=1 tp=1 pp=4 (diff=0.0)
  dp=2 tp=2 pp=1 (diff=1.19e-07)
  dp=2 tp=1 pp=2 (diff=0.0)
  dp=1 tp=2 pp=2 (diff=1.19e-07)
FAIL (0):
```

---

## Modal Cloud Training

Run distributed training on Modal with 4x A100 GPUs.

### Prerequisites

```bash
pip install modal
modal setup
```

Ensure you have these Modal resources configured:
- **Volume**: `fineweb-data` (with FineWeb10B data)
- **Secret**: `wandb-secret` (with `WANDB_API_KEY`)

### Run Training

```bash
modal run modal_train.py
```

### Default Configuration

| Setting | Value |
|---------|-------|
| GPUs | 4x A100-40GB |
| Model | GPT-2 Medium (24L/16H/1024D, ~355M params) |
| MoE | 8 experts, top-2, every 2 layers |
| Parallelism | DP=2, TP=2 |
| Batch | 8 per GPU, grad_accum=4 |
| Steps | 2000 |
| Precision | bf16 |

Results are logged to WandB project `moe-4d-parallel-training`.

### Estimated Cost

~$3-5 for 2000 steps (~25-30 minutes on 4x A100-40GB)

---

## Learning Plan

A comprehensive learning plan is available for mastering 4D parallelism using this codebase:

**[Learning Plan 4D Parallel/](Learning%20Plan%204D%20Parallel/)**

| Phase | Topic | Code File |
|-------|-------|-----------|
| 1 | Data Parallelism | `dp.py` |
| 2 | ZeRO Concepts | (theory) |
| 3 | Pipeline Parallelism | `pipeline.py` |
| 4 | Mixture of Experts | `moe.py` |
| 5 | Tensor Parallelism | `tp_linear.py` |
| 6 | Expert Parallelism | `ep_comm.py` |
| 7 | 4D Integration | `topo.py`, `train.py` |

---

## Notes

- PP uses `--grad_accum_steps` as the microbatch count
- Embedding/LM-head weight tying is only enabled when `pp=1`
- TP initialization uses per-rank seeds (SmolLM3 fix)
- MoE unused experts receive zero gradients to prevent NCCL hangs

---

## Acknowledgements

- **[Chinmay Kak](https://github.com/ChinmayK0607)** - Original author of [heiretsu](https://github.com/ChinmayK0607/heiretsu), the foundation for this project
- **[Scratch to Scale](https://maven.com/walk-with-code/scratch-to-scale)** course by Zachary Mueller
- **[nanochat](https://github.com/karpathy/nanochat)** by Andrej Karpathy - inspiration for minimal design
- **[Picotron](https://github.com/huggingface/picotron)** by HuggingFace - educational 4D parallelism
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** - TP patterns
- **SmolLM3 Training Playbook** - TP initialization bug fix
