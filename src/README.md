# Source Code

This folder is reserved for a cleaned public implementation if the source code is added later.

The current repository is portfolio-oriented and does not require source code to communicate the thesis work. If code is published, it should be reviewed and organized before being added.

## Recommended Future Structure

```text
src/
├── models/
│   ├── classical_autoencoder.py
│   ├── hybrid_autoencoder.py
│   ├── quantum_encoder.py
│   └── custom_quantum_encoder.py
├── channels/
│   ├── rayleigh.py
│   ├── rician.py
│   ├── gpp.py
│   └── urswipt.py
├── training/
│   ├── train_linear.py
│   └── train_urswipt.py
├── evaluation/
│   ├── evaluate_bler.py
│   └── evaluate_ser.py
├── plotting/
│   └── plot_results.py
└── utils/
```

## Before Publishing Code

Check for:

- Private file paths
- HPC-specific paths
- SLURM account names
- Large generated outputs
- Unnecessary checkpoints
- Private collaborator code
- Unused experimental scripts
- Hardcoded random seeds or undocumented settings
