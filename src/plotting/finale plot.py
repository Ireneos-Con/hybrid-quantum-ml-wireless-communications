import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ================================
# FILES
# ================================

# AE
AE_44_3GPP = BASE_DIR / "HAE44_3GPP.npz"
AE_44_RAY  = BASE_DIR / "hae44_Rayleigh.npz"
AE_44_RIC  = BASE_DIR / "HAE44_Rician.npz"

AE_74_3GPP = BASE_DIR / "HAE_74_vs_Hamming_3GPP.npz"
AE_74_RAY  = BASE_DIR / "hae74Rayleigh.npz"
AE_74_RIC  = BASE_DIR / "hae74Rician.npz"

AE_88_3GPP = BASE_DIR / "HAE_88_vs_BPSK_3GPP.npz"
AE_88_RAY  = BASE_DIR / "hae88_Rayleigh.npz"
AE_88_RIC  = BASE_DIR / "Hae88_Rician.npz"


# CAE (baseline μέσα εδώ)
CAE_44_3GPP = BASE_DIR / "CAE_44_vs_BPSK_3GPP_TR25996.npz"
CAE_44_RAY  = BASE_DIR / "CAE_44_vs_BPSK_Rayleigh(new).npz"
CAE_44_RIC  = BASE_DIR / "CAE_44_vs_BPSK_RicianK5_2.npz"

CAE_74_3GPP = BASE_DIR / "CAE_74_vs_Hamming_3GPP_TR25996.npz"
CAE_74_RAY  = BASE_DIR / "CAE_74vs_Hamming_Rayleigh.npz"
CAE_74_RIC  = BASE_DIR / "CAE_74_vs_Hamming_Rician.npz"

CAE_88_3GPP = BASE_DIR / "CAE_88_vs_BPSK_3GPP_TR25996.npz"
CAE_88_RAY  = BASE_DIR / "CAE_88_vs_BPSK_Rayleigh_2.npz"
CAE_88_RIC  = BASE_DIR / "CAE_88_vs_BPSK_Rician.npz"


# ================================
# LOAD
# ================================

def load_triplet(f3gpp, fray, fric):

    a = np.load(f3gpp)
    b = np.load(fray)
    c = np.load(fric)

    snr = a["snr_dBs"]

    return {
        "snr": snr,
        "3GPP": a,
        "Rayleigh": b,
        "Rician": c
    }


ae44 = load_triplet(AE_44_3GPP, AE_44_RAY, AE_44_RIC)
ae74 = load_triplet(AE_74_3GPP, AE_74_RAY, AE_74_RIC)
ae88 = load_triplet(AE_88_3GPP, AE_88_RAY, AE_88_RIC)

cae44 = load_triplet(CAE_44_3GPP, CAE_44_RAY, CAE_44_RIC)
cae74 = load_triplet(CAE_74_3GPP, CAE_74_RAY, CAE_74_RIC)
cae88 = load_triplet(CAE_88_3GPP, CAE_88_RAY, CAE_88_RIC)


AE_KEYS = {
    "44": {
        "3GPP": "bler_3GPP",
        "Rayleigh": "bler_ae_rayleigh",
        "Rician": "bler_RICIAN",
    },
    "74": {
        "3GPP": "bler_hae_3gpp",
        "Rayleigh": "bler_ae_rayleigh",
        "Rician": "bler_ae_RICIAN",
    },
    "88": {
        "3GPP": "bler_hae_3GPP",
        "Rayleigh": "bler_ae_rayleigh",
        "Rician": "bler_RICIAN",
    },
}

CAE_KEYS = {
    "44": {
        "3GPP": "bler_cae_3gpp_tr25996",
        "Rayleigh": "bler_cae_rayleigh",
        "Rician": "bler_cae_rician",
    },
    "74": {
        "3GPP": "bler_cae_3gpp_tr25996",
        "Rayleigh": "bler_cae_rayleigh",
        "Rician": "bler_cae_rician",
    },
    "88": {
        "3GPP": "bler_cae_3gpp_tr25996",
        "Rayleigh": "bler_cae_rayleigh",
        "Rician": "bler_cae_rician",
    },
}

BASELINE_KEYS = {
    "44": {
        "3GPP": "bler_bpsk_3gpp_tr25996",
        "Rayleigh": "bler_bpsk_rayleigh",
        "Rician": "bler_bpsk_rician",
    },
    "74": {
        "3GPP": "bler_hamming_3gpp_tr25996",
        "Rayleigh": "bler_hamming74_rayleigh",
        "Rician": "bler_hamming_rician",
    },
    "88": {
        "3GPP": "bler_bpsk_3gpp_tr25996",
        "Rayleigh": "bler_bpsk_rayleigh",
        "Rician": "bler_bpsk_rician",
    },
}


# ================================
# PLOT FUNCTION
# ================================

def plot_case(ax, ae, cae, title, case_id):
    method_colors = {
        "AE": "tab:blue",
        "CAE": "tab:orange",
        "Baseline": "tab:green",
    }

    channel_markers = {
        "3GPP": "o",
        "Rayleigh": "s",
        "Rician": "^",
    }

    for ch in ["3GPP", "Rayleigh", "Rician"]:
        snr = ae[ch]["snr_dBs"]

        ax.semilogy(
            snr,
            ae[ch][AE_KEYS[case_id][ch]],
            color=method_colors["AE"],
            marker=channel_markers[ch],
            linestyle="-",
        )

        ax.semilogy(
            snr,
            cae[ch][CAE_KEYS[case_id][ch]],
            color=method_colors["CAE"],
            marker=channel_markers[ch],
            linestyle="-",
        )

        ax.semilogy(
            snr,
            cae[ch][BASELINE_KEYS[case_id][ch]],
            color=method_colors["Baseline"],
            marker=channel_markers[ch],
            linestyle="-",
            label="Baseline" if ch == "3GPP" else None,
        )

    ax.set_title(title)
    ax.set_xlabel("SNR (dB)")
    ax.grid(True)
    ax.set_ylim(1e-4,1)


def plot_single_channel(channel_name):
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    cases = [
        ("44", ae44, cae44, "(a) n=4,k=4"),
        ("74", ae74, cae74, "(b) n=7,k=4"),
        ("88", ae88, cae88, "(c) n=8,k=8"),
    ]

    for ax, (case_id, ae, cae, title) in zip(axs, cases):
        snr = ae[channel_name]["snr_dBs"]

        ax.semilogy(
            snr,
            ae[channel_name][AE_KEYS[case_id][channel_name]],
            color="tab:blue",
            marker="o",
            linestyle="-",
            label="HAE",
        )
        ax.semilogy(
            snr,
            cae[channel_name][CAE_KEYS[case_id][channel_name]],
            color="tab:orange",
            marker="s",
            linestyle="-",
            label="CAE",
        )
        ax.semilogy(
            snr,
            cae[channel_name][BASELINE_KEYS[case_id][channel_name]],
            color="tab:green",
            marker="^",
            linestyle="-",
            label="Baseline",
        )

        ax.set_title(title)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True)
        ax.set_ylim(1e-4, 1)

    axs[0].set_ylabel("BLER")
    fig.suptitle(f"{channel_name} Only", fontsize=14, y=0.98)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=True)
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(BASE_DIR / f"{channel_name.upper()}.png", dpi=300, bbox_inches="tight")
    return fig


# ================================
# FIGURE
# ================================

fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

plot_case(axs[0], ae44, cae44, "(a) n=4,k=4", "44")
plot_case(axs[1], ae74, cae74, "(b) n=7,k=4", "74")
plot_case(axs[2], ae88, cae88, "(c) n=8,k=8", "88")

axs[0].set_ylabel("BLER")

method_handles = [
    Line2D([0], [0], color="tab:blue", marker=None, linestyle="-", label="HAE"),
    Line2D([0], [0], color="tab:orange", marker=None, linestyle="-", label="CAE"),
    Line2D([0], [0], color="tab:green", marker=None, linestyle="-", label="Baseline"),
]

channel_handles = [
    Line2D([0], [0], color="black", marker="o", linestyle="None", label="3GPP"),
    Line2D([0], [0], color="black", marker="s", linestyle="None", label="Rayleigh"),
    Line2D([0], [0], color="black", marker="^", linestyle="None", label="Rician"),
]

legend_methods = fig.legend(
    handles=method_handles,
    loc="upper center",
    bbox_to_anchor=(0.28, 1.02),
    ncol=3,
    frameon=True,
    title="Method",
)

legend_channels = fig.legend(
    handles=channel_handles,
    loc="upper center",
    bbox_to_anchor=(0.77, 1.02),
    ncol=3,
    frameon=True,
    title="Channel",
)

fig.add_artist(legend_methods)
fig.add_artist(legend_channels)

plt.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(BASE_DIR / "FINAL_ALL_CHANNELS_CAE_TR25996_3GPP.png", dpi=300, bbox_inches="tight")

plot_single_channel("Rayleigh")
plot_single_channel("Rician")
plot_single_channel("3GPP")

plt.show()
