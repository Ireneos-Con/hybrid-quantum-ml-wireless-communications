import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
P12_PATH = BASE_DIR / "12P.npz"
HAE74_PATH = BASE_DIR / "hae74.npz"
OUTPUT_PATH = BASE_DIR / "p12_vs_hae74_rayleigh.png"


def load_results(npz_path):
    data = np.load(npz_path)
    return data["snr_dBs"], data["bler_ae_rayleigh"]


def main():
    snr_p12, bler_p12 = load_results(P12_PATH)
    snr_hae, bler_hae = load_results(HAE74_PATH)

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_p12, bler_p12, "o-", linewidth=2, markersize=6, label="P12")
    plt.semilogy(snr_hae, bler_hae, "s--", linewidth=2, markersize=6, label="Original HAE74")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title("Rayleigh BLER Comparison: P12 vs Original HAE74")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
