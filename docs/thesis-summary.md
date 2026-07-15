# Thesis Summary

## Title

Hybrid Quantum-Classical Machine Learning for Autoencoder-Based Wireless Communication Systems

## Author

Ireneos Constantinou  
Department of Electrical and Computer Engineering  
University of Cyprus  
Advisor: Dr Ioannis Krikides  
May 2026

## Abstract Summary

This thesis studies hybrid quantum-classical machine learning for end-to-end wireless communication systems modeled as autoencoders. Classical autoencoder-based communication systems can jointly optimize transmitter and receiver behavior, but as the number of messages and model dimensions grow, the number of trainable parameters can increase significantly.

The central objective is to examine whether hybrid quantum-classical autoencoders can reduce trainable parameters while maintaining performance comparable to classical autoencoder architectures. The hybrid model replaces the classical encoder with a quantum encoder based on parameterized quantum circuits, while the decoder remains classical.

The evaluation compares hybrid autoencoders (HAEs) and classical autoencoders (CAEs) across linear fading channels and a nonlinear UR-SWIPT communication setting. The results show that hybrid models can achieve comparable BLER and SER behavior while using significantly fewer trainable parameters.

## Main Contributions

- Reproduced and analyzed CAE vs HAE behavior for end-to-end wireless communication systems.
- Evaluated the models across Rayleigh, Rician, and 3GPP fading channels.
- Quantified parameter savings across multiple (n, k) configurations.
- Designed and evaluated lower-parameter PQC variants, including P12 and P18.
- Developed a custom quantum encoder using angle embedding, adaptive layer scaling, and readout control.
- Extended the hybrid approach to the nonlinear UR-SWIPT information decoding setting.
- Used HPC infrastructure and SLURM workflows for model training and evaluation.

## Core Idea

The classical autoencoder uses neural-network layers to map messages into transmitted representations and decode received signals. The hybrid autoencoder keeps the end-to-end learning structure but replaces the encoder with a quantum circuit. The quantum encoder uses classical-to-quantum embedding, trainable rotations, entanglement, and measurements to produce signal features that are passed through the communication channel and then decoded classically.

## Evaluated Systems

- Classical autoencoder (CAE)
- Hybrid quantum-classical autoencoder (HAE)
- P12 low-parameter PQC variant
- P18 low-parameter PQC variant
- Custom quantum encoder
- Hybrid UR-SWIPT information decoding model

## Channels and Metrics

The thesis evaluates performance using:

- Rayleigh fading
- Rician fading
- 3GPP fading
- Nonlinear UR-SWIPT channel
- Block Error Rate (BLER)
- Symbol Error Rate (SER)
- Signal-to-Noise Ratio (SNR)
- Trainable parameter count

## Key Results

- For the linear-channel experiments, the HAE reduced trainable parameters by approximately 60-70% compared with the CAE.
- In the (4, 4), (7, 4), and (8, 8) linear-channel settings, the HAE remained close to the CAE in BLER vs SNR performance.
- The P18 encoder reduced quantum encoder parameters while preserving most of the original HAE performance.
- The custom encoder achieved a BLER trend close to the reference (7, 4) HAE Rayleigh model.
- In the M = 8 UR-SWIPT setup, the HAE reduced total trainable parameters from 169 to 106, about 37.3%.

## Conclusion

The thesis shows that hybrid quantum-classical autoencoders are a promising direction for parameter-efficient wireless communication system design. Although quantum circuits were simulated on classical hardware, limiting practical computational speedups, the reduction in trainable parameters and competitive BLER/SER performance show that hybrid models can offer compact alternatives to purely classical autoencoder architectures.
