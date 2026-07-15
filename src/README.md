# Source Code

This folder contains selected thesis experiment scripts organized as a public research archive.

The code is not presented as a polished Python package. It is included to show the main implementation work behind the thesis: classical autoencoder experiments, hybrid quantum-classical autoencoder experiments, UR-SWIPT experiments, and plotting scripts.

## Layout

```text
src/
├── experiments/
│   ├── classical/
│   │   ├── rayleigh/
│   │   ├── rician/
│   │   └── 3gpp/
│   ├── hybrid/
│   │   ├── hae-4-4/
│   │   ├── hae-7-4/
│   │   └── hae-8-8/
│   └── urswipt/
│       └── ptfull/
└── plotting/
```

## Included Experiment Groups

- `experiments/classical/`: CAE scripts for Rayleigh, Rician, and 3GPP channel evaluations.
- `experiments/hybrid/`: HAE scripts for the (4, 4), (7, 4), and (8, 8) settings, including low-parameter variants where relevant.
- `experiments/urswipt/`: UR-SWIPT information decoding scripts.
- `plotting/`: scripts used to combine final result files and generate thesis figures.

## Excluded Files

The following were intentionally excluded from the public repository:

- `.out` and `.err` HPC logs
- `__pycache__` folders
- `.pth` model checkpoints
- temporary or duplicate generated files
- machine-specific SLURM scripts containing local HPC paths
- document drafts and unrelated support files

## Notes

Some scripts were originally developed for local and HPC experimentation, so paths and execution details may need adjustment before running in a different environment. The thesis PDF and documentation should be treated as the primary source of the exact experimental setup.
