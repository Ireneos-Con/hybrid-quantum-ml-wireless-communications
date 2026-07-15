# Reproduction Notes

This repository is currently portfolio-oriented. The full source implementation is not included in the current public version.

## Current Public Artifacts

- Full thesis PDF
- Research poster PDF
- Thesis summary
- Methodology notes
- Results summary
- Optional source-code folder reserved for future release

## Implementation Status

The thesis implementation used a Python-based ML/QML workflow with PyTorch and PennyLane. Experiments were developed locally and executed on HPC infrastructure through SLURM batch jobs.

The code can be added later after cleanup. Before publishing source code, the following should be reviewed:

- Private paths
- University HPC paths
- SLURM account or partition information
- Generated logs
- Large result files
- Checkpoints
- Unpublished collaborator code
- Any private research dependencies

## Expected Environment

If the implementation is added later, the expected stack is:

```text
Python 3.10+
PyTorch
PennyLane
PennyLane Lightning
NumPy
SciPy
Matplotlib
SLURM for HPC execution
```

## Suggested Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux or HPC environments:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested Future Commands

The exact commands depend on the final public code structure. A clean implementation could expose commands such as:

```bash
python src/train_linear.py --model hae --channel rayleigh --n 7 --k 4
python src/evaluate_linear.py --model hae --channel rayleigh --n 7 --k 4
python src/train_urswipt.py --model hae --M 8
python src/plot_results.py
```

## Public Release Recommendation

For a portfolio repository, it is acceptable to keep the implementation private and document the research clearly. If code is added, publish only a cleaned minimal version that reproduces the main figures or demonstrates the model architecture.
