# Hybrid Quantum-Classical Machine Learning for Autoencoder-Based Wireless Communication Systems

Portfolio repository for my completed diploma thesis at the Department of Electrical and Computer Engineering, University of Cyprus.

The thesis investigates hybrid quantum-classical machine learning models for end-to-end autoencoder-based wireless communication systems. The main focus is parameter efficiency: whether a hybrid autoencoder (HAE), using a quantum encoder and a classical decoder, can reduce the number of trainable parameters while maintaining communication performance close to a classical autoencoder (CAE).

## Thesis Artifacts

- [Full thesis PDF](thesis/hybrid-quantum-classical-ml-autoencoder-wireless-communication-systems.pdf)
- [Research poster](presentation/ireneos-poster.pdf)
- [Thesis summary](docs/thesis-summary.md)
- [Methodology](docs/methodology.md)
- [Reproduction notes](docs/reproduction-notes.md)

## Project Overview

Machine learning has become increasingly important in wireless communications, especially through end-to-end learning architectures where the transmitter and receiver are modeled jointly as an autoencoder. As these systems scale, classical models can require large numbers of trainable parameters, increasing memory requirements and model complexity.

This thesis studies hybrid quantum-classical autoencoders as a compact alternative. The encoder is implemented using parameterized quantum circuits (PQCs), while the decoder remains classical. The system is evaluated across linear fading channels and a nonlinear UR-SWIPT communication setting.

## Research Questions

- Can a hybrid quantum-classical autoencoder reduce trainable parameters while preserving BLER performance close to a classical autoencoder?
- How does the hybrid model behave across Rayleigh, Rician, and 3GPP fading channels?
- Can the quantum encoder be compressed further through parameter-sharing variants such as P12 and P18?
- Can a custom quantum encoder remain competitive while staying within a controlled parameter budget?
- Does the hybrid approach remain useful in nonlinear wireless communication channels such as UR-SWIPT?

## Key Findings

- The HAE achieved approximately 60-70% parameter reduction compared with the CAE in the tested linear-channel configurations.
- Across Rayleigh, Rician, and 3GPP channels, the HAE achieved BLER performance close to the CAE while using significantly fewer trainable parameters.
- The P18 low-parameter quantum encoder preserved most of the original HAE performance, while P12 introduced more visible degradation.
- A custom quantum encoder using angle embedding, adaptive layer scaling, and readout control achieved BLER behavior close to the reference HAE in the (7, 4) Rayleigh setting.
- In the nonlinear UR-SWIPT information decoding task, the HAE reduced trainable parameters by approximately 37.3% while maintaining SER performance close to the learned classical and algorithmic references.

## Methodology

The work compares classical and hybrid quantum-classical autoencoder architectures for wireless communication.

The evaluation includes:

- CAE vs HAE over linear fading channels
- Rayleigh, Rician, and 3GPP channel models
- BLER vs SNR evaluation
- Trainable parameter comparison
- Low-parameter quantum encoder variants P12 and P18
- A custom quantum encoder with a lightweight classical controller
- Hybrid UR-SWIPT evaluation for nonlinear wireless communication
- HPC-based training using SLURM and PennyLane's `lightning.qubit` simulator

## Results Summary

| Experiment | Main Result |
| --- | --- |
| Linear channels | HAE achieved comparable BLER to CAE with around 60-70% fewer trainable parameters. |
| Rayleigh channel | HAE stayed close to CAE in the (4, 4), (7, 4), and (8, 8) settings. |
| Rician channel | HAE remained close to CAE, with the (7, 4) HAE slightly outperforming CAE at high SNR in the reported evaluation. |
| 3GPP channel | HAE and CAE showed similar BLER behavior, especially in the (7, 4) setting. |
| P12/P18 variants | P18 was a balanced low-parameter design; P12 compressed more aggressively but degraded more. |
| Custom encoder | Reached BLER trends close to the reference (7, 4) HAE Rayleigh implementation. |
| UR-SWIPT | HAE reduced parameters from 169 to 106 in the M = 8 setup, a reduction of about 37.3%. |

## Technical Stack

- Python
- PyTorch
- PennyLane
- PennyLane `lightning.qubit`
- NumPy
- SciPy
- Matplotlib
- SLURM
- University HPC infrastructure

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── docs/
│   ├── thesis-summary.md
│   ├── methodology.md
│   └── reproduction-notes.md
├── presentation/
│   ├── README.md
│   └── ireneos-poster.pdf
├── thesis/
│   ├── README.md
│   └── hybrid-quantum-classical-ml-autoencoder-wireless-communication-systems.pdf
├── results/
│   ├── README.md
│   ├── figures/
│   └── tables/
├── notebooks/
│   └── README.md
└── src/
    └── README.md
```

## Source Code Availability

The current version of this repository is portfolio-oriented and focuses on the thesis, poster, methodology, and results. Source code may be added later after review and cleanup.

If the implementation is not included, this repository still documents the research objective, model design, evaluation setup, and main findings.

## Citation

If you reference this work, please use the metadata in [CITATION.cff](CITATION.cff).

## Author

Ireneos Constantinou  
Department of Electrical and Computer Engineering  
University of Cyprus  
May 2026
