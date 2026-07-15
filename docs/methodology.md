# Methodology

This document summarizes the research methodology used in the thesis.

## Problem Definition

The thesis studies end-to-end wireless communication systems where the transmitter and receiver are modeled jointly as an autoencoder. The main problem is model complexity: as autoencoder-based communication systems scale, the number of trainable parameters can grow significantly.

The research investigates whether a hybrid quantum-classical model can reduce the number of trainable parameters while maintaining communication reliability.

## Classical Autoencoder Baseline

The classical autoencoder (CAE) represents the transmitter as a neural-network encoder and the receiver as a neural-network decoder. A source message is encoded into a transmitted representation, passed through a channel model, and decoded back into an estimated message.

The CAE is used as the baseline for performance and parameter-count comparison.

## Hybrid Autoencoder Architecture

The hybrid autoencoder (HAE) keeps the same end-to-end communication framework, but replaces the classical encoder with a quantum encoder.

The HAE uses:

- Amplitude embedding for classical message representation in the baseline quantum encoder
- Parameterized quantum circuits (PQCs)
- Trainable Ry rotations
- Nearest-neighbor CNOT entanglement
- Pauli-Z measurements
- Separate real and imaginary branches
- Power normalization before channel transmission
- A classical decoder at the receiver

## Low-Parameter Quantum Encoder Variants

Two lower-parameter PQC variants were studied for the (7, 4) setting.

### P12 Variant

The P12 model uses parity-based parameter sharing. Even-indexed qubits share one trainable rotation parameter and odd-indexed qubits share another parameter within each layer.

This gives:

```text
3 layers x 2 groups x 2 branches = 12 trainable quantum encoder parameters
```

### P18 Variant

The P18 model uses three fixed qubit groups:

```text
G1 = {0, 3, 6}
G2 = {1, 4}
G3 = {2, 5}
```

This gives:

```text
3 layers x 3 groups x 2 branches = 18 trainable quantum encoder parameters
```

P18 was less aggressive than P12 and preserved more of the original HAE behavior.

## Custom Quantum Encoder

A custom quantum encoder was also designed. Instead of amplitude embedding, it uses angle embedding based on the binary representation of each transmitted message.

The design includes:

- Bit-to-angle mapping through Rx rotations
- A trainable PQC with Ry rotations and CNOT entanglement
- A lightweight classical controller
- Symbol-dependent layer scaling
- Symbol-dependent readout rotation before Pauli-Z measurement
- Separate real and imaginary quantum branches
- Output stacking and power normalization

The custom encoder was evaluated in the (7, 4) Rayleigh setting and showed BLER behavior close to the reference hybrid implementation.

## Nonlinear UR-SWIPT Setting

The thesis also studies a nonlinear communication scenario using Unified Receiver Simultaneous Wireless Information and Power Transfer (UR-SWIPT).

In UR-SWIPT, the received signal is processed through a rectifying receiver, making the effective channel nonlinear. The hybrid implementation replaces the encoder with a quantum-based architecture while keeping the rectifier model and decoder in the classical domain.

The UR-SWIPT evaluation focuses on information decoding (ID) performance and parameter savings.

## Training and Evaluation Setup

The linear-channel experiments use:

- Batch size: 32
- Epochs: 80
- Training SNR: 10 dB
- Optimizer: Adam
- Learning rate: 0.001
- Evaluation SNR range: 0 to 20 dB
- Monte Carlo evaluation
- 75,000 generated test samples per SNR point

The primary evaluation metrics are:

- BLER for linear-channel autoencoder models
- SER for UR-SWIPT information decoding
- Trainable parameter count

## Computational Infrastructure

Experiments were developed locally and executed through the University of Cyprus HPC infrastructure.

The workflow used:

- GitHub for version control and transfer to the HPC environment
- SLURM job submission
- CPU compute nodes
- PyTorch for classical neural-network components
- PennyLane for quantum circuits
- `lightning.qubit` as the quantum simulation backend
- Saved `.npz`, `.out`, and `.err` outputs for later plotting

The thesis notes that `lightning.gpu` would be suitable for GPU-accelerated execution, but the available infrastructure used CPU-based `lightning.qubit`.
