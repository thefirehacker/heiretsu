# Learning Plan: Mastering 4D Parallelism

## Overview

| Field | Details |
|-------|---------|
| **Objective** | Master Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP), and Expert Parallelism (EP) through hands-on implementation |
| **Course** | [Scratch to Scale](https://maven.com/walk-with-code/scratch-to-scale) by Zachary Mueller |
| **Codebase** | [heiretsu](.) - Minimal 3D/4D parallelism in pure PyTorch |
| **Duration** | 4 weeks (matching course) + bonus content |
| **Prerequisites** | PyTorch basics, single-GPU training experience, basic Python |

---

## Learning Philosophy

This plan bridges **theory** (course lectures) with **practice** (heiretsu codebase):

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Watch Lecture  │ ──► │  Read Code File  │ ──► │   Run Test      │
│  (understand)   │     │  (see impl)      │     │   (verify)      │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Modify & Experiment   │
                    │  (deepen mastery)      │
                    └────────────────────────┘
```

---

## Phase Structure

| Phase | Name | Week | Focus | Key File |
|-------|------|------|-------|----------|
| 1 | **Data Parallelism** | 1 | Gradient sync, batch sharding | `dp.py` |
| 2 | **ZeRO Concepts** | 2 | Memory optimization (theory) | Course only |
| 3 | **Pipeline Parallelism** | 3 | Microbatching, stage communication | `pipeline.py` |
| 4 | **Mixture of Experts** | 3 | Routing, load balancing | `moe.py` |
| 5 | **Tensor Parallelism** | 4 | Column/row sharding, TP init | `tp_linear.py` |
| 6 | **Expert Parallelism** | Bonus | All-to-All dispatch | `ep_comm.py` |
| 7 | **4D Integration** | Bonus | Combining all methods | `topo.py`, `train.py` |

---

## 4D Parallelism Concept Map

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  WORLD (8 GPUs)                     │
                    │   ┌───────────────────┬───────────────────┐         │
                    │   │     DP Rank 0     │     DP Rank 1     │         │
                    │   │   (batch shard)   │   (batch shard)   │         │
                    │   │   ┌─────┬─────┐   │   ┌─────┬─────┐   │         │
                    │   │   │TP=0 │TP=1 │   │   │TP=0 │TP=1 │   │         │
                    │   │   │(col)│(col)│   │   │(col)│(col)│   │  TP     │
                    │   │   ├─────┼─────┤   │   ├─────┼─────┤   │ (weight │
                    │   │   │PP=0 │PP=0 │   │   │PP=0 │PP=0 │   │  shard) │
                    │   │   │(L0-5)│    │   │   │(L0-5)│    │   │         │
                    │   │   ├─────┼─────┤   │   ├─────┼─────┤   │  PP     │
                    │   │   │PP=1 │PP=1 │   │   │PP=1 │PP=1 │   │ (layer  │
                    │   │   │(L6-11)   │   │   │(L6-11)   │   │  stages) │
                    │   │   └─────┴─────┘   │   └─────┴─────┘   │         │
                    │   └───────────────────┴───────────────────┘         │
                    │                                                     │
                    │   EP: Experts distributed across DP or dedicated    │
                    │       ranks for MoE layers                          │
                    └─────────────────────────────────────────────────────┘
```

**Rank Layout Formula** (from `topo.py`):
```
global_rank = dp*(EP*PP*TP) + ep*(PP*TP) + pp*TP + tp
```

---

## Codebase Architecture

```
heiretsu/
├── train.py          # Main entry: training loop, integrates all parallelism
├── gpt_model.py      # Model: GPT with TP/MoE support
├── topo.py           # Topology: process group management (DP/EP/TP/PP)
├── dp.py             # Data Parallelism: gradient averaging
├── tp_linear.py      # Tensor Parallelism: Column/Row parallel layers
├── pipeline.py       # Pipeline Parallelism: GPipe microbatching
├── moe.py            # MoE: Router, Experts, load balancing
├── ep_comm.py        # Expert Parallelism: All-to-All communication
└── tests/            # Verification tests for each parallelism type
```

---

## GPU Requirements

| Test Level | GPUs Needed | Example Configs |
|------------|-------------|-----------------|
| **Beginner** | 1-2 | Single, DP=2 |
| **Intermediate** | 2-4 | TP=2, PP=2, DP+TP |
| **Advanced** | 4-8 | DP=2+TP=2+PP=2, MoE with EP |

---

## Quick Verification Commands

Test your setup before starting:

```bash
# Single GPU baseline (no dist required)
python tests/forward_single.py

# 2-GPU DP test
torchrun --standalone --nproc_per_node=2 tests/forward_dp.py

# Full test suite (4+ GPUs)
bash tests/run_full_suite.sh
```

---

## Resources

| Resource | Description |
|----------|-------------|
| **Distributed Operations Cheatsheet** | PDF with visuals of torch.distributed ops |
| **Distributed Training Lexicon** | 49 terms with definitions |
| **Course Discord** | Community support (lifetime access) |
| `README.md` | Codebase quick start and test commands |

---

## Navigation

- [00-Overview.md](00-Overview.md) - You are here
- [01-Phase-DP.md](01-Phase-DP.md) - Data Parallelism
- [02-Phase-ZeRO.md](02-Phase-ZeRO.md) - ZeRO Concepts (theory)
- [03-Phase-PP.md](03-Phase-PP.md) - Pipeline Parallelism
- [04-Phase-MoE.md](04-Phase-MoE.md) - Mixture of Experts
- [05-Phase-TP.md](05-Phase-TP.md) - Tensor Parallelism
- [06-Phase-EP.md](06-Phase-EP.md) - Expert Parallelism
- [07-Phase-Integration.md](07-Phase-Integration.md) - 4D Integration
- [Concept-Mappings.md](Concept-Mappings.md) - Theory-to-code bridges
- [Progress-Tracker.md](Progress-Tracker.md) - Checklist for tracking
