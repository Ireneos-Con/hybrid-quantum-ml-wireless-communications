# Results

This directory is reserved for result figures and tables from the thesis.

The full thesis PDF already contains the complete figures and tables. Selected exported figures can be added here later for easier viewing directly on GitHub.

## Main Reported Results

### Linear Channels: CAE vs HAE

The HAE consistently used fewer trainable parameters than the CAE:

| Configuration | CAE Parameters | HAE Parameters |
| --- | ---: | ---: |
| (4, 4) | 1232 | 440 |
| (7, 4) | 1532 | 554 |
| (8, 8) | 209952 | 70192 |

This corresponds to approximately 60-70% fewer trainable parameters.

### Fading Channel Performance

The BLER vs SNR evaluation showed that the HAE remained close to the CAE across:

- Rayleigh fading
- Rician fading
- 3GPP fading

For example, in the Rayleigh channel at 20 dB, the HAE remained close to the CAE in the tested (4, 4), (7, 4), and (8, 8) settings.

### Low-Parameter PQC Variants

| Model Variant | Quantum Encoder Params | Decoder Params | Total Params |
| --- | ---: | ---: | ---: |
| Original quantum encoder | 42 | 512 | 554 |
| P12 low-parameter PQC | 12 | 512 | 524 |
| P18 low-parameter PQC | 18 | 512 | 530 |

P18 preserved most of the original HAE performance, while P12 showed more visible degradation.

### Custom Quantum Encoder

The custom encoder used:

- Angle embedding
- Adaptive layer scaling
- Symbol-dependent readout rotation
- Two quantum branches for real and imaginary components

It achieved BLER behavior close to the reference (7, 4) HAE Rayleigh implementation.

### UR-SWIPT

For the M = 8 UR-SWIPT information decoding setup:

| Scheme | Encoder Params | Decoder Params | Total Params |
| --- | ---: | ---: | ---: |
| CAE | 81 | 88 | 169 |
| HAE | 18 | 88 | 106 |

This corresponds to approximately 37.3% parameter reduction while maintaining SER performance close to the learned classical and algorithmic references.

## Suggested Future Figure Exports

Export the following figures from the thesis into `results/figures/`:

- `hae-vs-cae-rayleigh.png`
- `hae-vs-cae-rician.png`
- `hae-vs-cae-3gpp.png`
- `p18-vs-original-hae.png`
- `custom-encoder-bler-vs-snr.png`
- `urswipt-ser-vs-snr.png`

## Suggested Future Tables

Export the following tables into `results/tables/`:

- `linear-parameter-comparison.csv`
- `pqc-variant-parameter-comparison.csv`
- `urswipt-parameter-comparison.csv`
