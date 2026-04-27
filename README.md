# HyPE: Hybrid Private Inference Framework
Artifact for CCS 2026 Submission.

## Structure
- /compiler: ONNX partitioning logic.
- /runtime: SGX and MPC backend integration.
- /data: Secret sharing utilities.

## Setup
1. Install Intel SGX SDK v2.12.
2. Install CrypTFlow2.
3. Run `python compiler/partitioner.py` to prepare the model.
