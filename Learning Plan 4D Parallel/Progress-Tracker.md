# Progress Tracker

Track your learning progress through the 4D Parallelism plan.

---

## Overall Status

| Phase | Name | Status | Started | Completed |
|-------|------|--------|---------|-----------|
| 1 | Data Parallelism | ⬜ Not Started | | |
| 2 | ZeRO Concepts | ⬜ Not Started | | |
| 3 | Pipeline Parallelism | ⬜ Not Started | | |
| 4 | Mixture of Experts | ⬜ Not Started | | |
| 5 | Tensor Parallelism | ⬜ Not Started | | |
| 6 | Expert Parallelism | ⬜ Not Started | | |
| 7 | 4D Integration | ⬜ Not Started | | |

**Legend:** ⬜ Not Started | 🟡 In Progress | ✅ Completed

---

## Phase 1: Data Parallelism

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: DDP from Scratch lecture
- [ ] Watched: DataLoader Workshop
- [ ] Understand: Batch sharding across GPUs
- [ ] Understand: Gradient averaging with all-reduce

### Practice (Code)
- [ ] Read: `dp.py` completely (32 lines)
- [ ] Read: `topo.py` - DP group creation
- [ ] Traced: Gradient flow through `average_gradients()`
- [ ] Found: Where `average_gradients` is called in `train.py`

### Verification (Tests)
- [ ] Ran: `python tests/forward_single.py`
- [ ] Ran: `torchrun --nproc=2 tests/forward_dp.py`
- [ ] Verified: `diff_max = 0.0` (identical logits)
- [ ] Ran: `bash tests/compare_dp.sh`

### Exercises
- [ ] Completed: Exercise 1 - Verify Gradient Sync
- [ ] Completed: Exercise 2 - Break DP
- [ ] Completed: Exercise 3 - Manual Gradient Check

### Notes
```
[Your notes here]
```

---

## Phase 2: ZeRO Concepts

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: ZeRO Stage 1 & 2 lecture
- [ ] Watched: ZeRO Stage 3 lecture
- [ ] Watched: PyTorch FSDP lecture
- [ ] Watched: Sylvain Gugger's ZeRO overview

### Understanding
- [ ] Can explain: ZeRO Stage 1 (optimizer sharding)
- [ ] Can explain: ZeRO Stage 2 (gradient sharding)
- [ ] Can explain: ZeRO Stage 3 (parameter sharding)
- [ ] Can compare: ZeRO-3 vs TP tradeoffs

### Knowledge Check
- [ ] Answered: Q1 - Which ZeRO stage for 7B on 8 GPUs?
- [ ] Answered: Q2 - Why TP preferred over ZeRO-3 for inference?

### Notes
```
[Your notes here]
```

---

## Phase 3: Pipeline Parallelism

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: Pipeline Parallelism lecture (1.5 hrs)
- [ ] Watched: PP with torchtitan lecture
- [ ] Understand: Microbatching and bubble overhead
- [ ] Know: Bubble formula = (P-1)/M

### Practice (Code)
- [ ] Read: `_partition_layers()` in `pipeline.py`
- [ ] Read: `StageModule` class
- [ ] Read: `_p2p()` method
- [ ] Traced: Forward pass through `forward_backward()`
- [ ] Traced: Backward gradient flow

### Verification (Tests)
- [ ] Ran: `torchrun --nproc=2 tests/forward_pp.py --pp 2`
- [ ] Ran: `bash tests/compare_pp.sh`
- [ ] Ran: `torchrun --nproc=4 tests/parallel_sanity.py --pp 4`

### Exercises
- [ ] Completed: Exercise 1 - Trace Boundary Tensors
- [ ] Completed: Exercise 2 - Verify Gradient Flow
- [ ] Completed: Exercise 3 - Bubble Calculation

### Notes
```
[Your notes here]
```

---

## Phase 4: Mixture of Experts

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: How to train your small MoE lecture
- [ ] Watched: Elie Bakouch's MoE optimization lecture
- [ ] Understand: Top-K routing
- [ ] Understand: Load balancing loss purpose

### Practice (Code)
- [ ] Read: `Router` class in `moe.py`
- [ ] Read: `load_balancing_loss()` function
- [ ] Read: `Expert` and `ExpertGroup` classes
- [ ] Traced: Full `MoELayer.forward()` shape flow

### Verification (Tests)
- [ ] Ran: `python tests/test_moe.py`
- [ ] Ran: Training with MoE enabled
- [ ] Verified: Loss decreases with MoE

### Exercises
- [ ] Completed: Exercise 1 - Visualize Expert Load
- [ ] Completed: Exercise 2 - Break Load Balancing
- [ ] Completed: Exercise 3 - Trace Permutation

### Notes
```
[Your notes here]
```

---

## Phase 5: Tensor Parallelism

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: Tensor Parallelism lecture (1.5 hrs)
- [ ] Watched: TP with TorchTitan lecture
- [ ] Watched: Less Wright's Async TP lecture
- [ ] Watched: Wanchao Liang's TorchTitan intro

### Practice (Code)
- [ ] Read: Autograd-aware collectives (lines 19-61)
- [ ] Read: `ColumnParallelLinear` class
- [ ] Read: `RowParallelLinear` class
- [ ] Read: `reset_parameters()` - SmolLM3 fix
- [ ] Found: TP usage in `gpt_model.py`

### Verification (Tests)
- [ ] Ran: `torchrun --nproc=2 tests/forward_tp.py --tp 2`
- [ ] Ran: `torchrun --nproc=2 tests/tests_equiv.py --tp 2`
- [ ] Ran: `bash tests/compare_tp.sh`
- [ ] Ran: `torchrun --nproc=4 tests/parallel_sanity.py --dp 2 --tp 2`

### Exercises
- [ ] Completed: Exercise 1 - Verify Weight Sharding
- [ ] Completed: Exercise 2 - Trace All-Reduce
- [ ] Completed: Exercise 3 - Break TP Init (SmolLM3 bug)

### Notes
```
[Your notes here]
```

---

## Phase 6: Expert Parallelism

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: Expert Parallelism lecture (Matej Sirovatka)
- [ ] Watched: Multi-dimensional parallelism (Ferdinand Mom)
- [ ] Understand: All-to-All communication
- [ ] Understand: How EP complements MoE

### Practice (Code)
- [ ] Read: EP group creation in `topo.py`
- [ ] Read: `all_to_all_dispatch()` in `ep_comm.py`
- [ ] Read: `all_to_all_combine()` in `ep_comm.py`
- [ ] Traced: EP path in `MoELayer.forward()`

### Verification (Tests)
- [ ] Ran: Full test suite with EP configs (8 GPUs)
- [ ] Ran: DP+EP configuration manually

### Exercises
- [ ] Completed: Exercise 1 - Trace Token Movement
- [ ] Completed: Exercise 2 - Verify Expert Ownership
- [ ] Completed: Exercise 3 - Measure All-to-All Latency

### Notes
```
[Your notes here]
```

---

## Phase 7: 4D Integration

**Started:** _____ | **Completed:** _____

### Theory (Course)
- [ ] Watched: Multi-dimensional parallelism (Ferdinand Mom)
- [ ] Watched: 2D Parallelism with Axolotl (Wing Lian)
- [ ] Watched: Backprop from Scratch (Daniel Han)
- [ ] Watched: DiLoCo (Sami Jaghouar)

### Practice (Code)
- [ ] Mastered: Topology rank layout formula
- [ ] Understood: Process group relationships
- [ ] Traced: Complete training step through `train.py`
- [ ] Read: `modal_train.py` for cloud deployment

### Verification (Tests)
- [ ] Ran: Full test suite (4+ GPUs)
- [ ] Ran: Full 3D test (8 GPUs): `dp=2 tp=2 pp=2`
- [ ] Ran: 4D with MoE (8 GPUs)
- [ ] Deployed: Training to Modal cloud

### Exercises
- [ ] Completed: Exercise 1 - Visualize Rank Topology
- [ ] Completed: Exercise 2 - Profile Communication
- [ ] Completed: Exercise 3 - Compare Configurations

### Final Project
- [ ] Completed: Project 1, 2, or 3

### Notes
```
[Your notes here]
```

---

## Graduation Checklist

Complete these to confirm mastery:

- [ ] Can explain each parallelism type in one sentence
- [ ] Can implement gradient averaging (DP)
- [ ] Can trace column/row parallel layers (TP)
- [ ] Can debug pipeline deadlocks (PP)
- [ ] Can configure MoE with load balancing
- [ ] Can understand All-to-All dispatch (EP)
- [ ] Can deploy multi-GPU training to cloud
- [ ] Can diagnose common distributed training failures

---

## Test Results Log

| Date | Test | Config | Result | Notes |
|------|------|--------|--------|-------|
| | | | | |
| | | | | |
| | | | | |

---

## Learning Notes

### Key Insights
```
[Record important realizations here]
```

### Confusing Parts
```
[Note areas that need review]
```

### Questions to Research
```
[Questions that arose during learning]
```

---

## Time Log

| Date | Phase | Hours | Activity |
|------|-------|-------|----------|
| | | | |
| | | | |
| | | | |

**Total Hours:** _____
