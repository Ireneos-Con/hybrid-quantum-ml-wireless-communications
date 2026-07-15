import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pennylane as qml
from scipy.special import i0
from scipy.special import lambertw
from scipy.optimize import root_scalar
from scipy.special import erfc

# =========================================================
# device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# system parameters
# =========================================================
k   = 2
M   = 4
R   = 8.25e3
eta = 1.05
Vt  = 25.85e-3
Is  = 5e-6
Rin = 50
K   = (Is * R) / (eta * Vt)

noise_std = 2e-2   # fixed evaluation noise (same as your file)

# =========================================================
# Lambert W torch (same as training)
# =========================================================
def lambertw_torch(x, iters=8):
    w = torch.log1p(x)
    for _ in range(iters):
        ew = torch.exp(w)
        f  = w * ew - x
        w  = w - f / (ew * (w + 1) + 1e-8)
    return w

# =========================================================
# Rectifier (IDENTICAL to training)
# =========================================================
class RectifierEH(nn.Module):
    def __init__(self, eta, Vt, K, Rin, Pt):
        super().__init__()
        self.eta     = eta
        self.Vt      = Vt
        self.K       = K
        self.eK      = math.exp(K)
        self.sqrtRin = math.sqrt(Rin)
        self.sqrtPt  = math.sqrt(2 * Pt)

    def forward(self, y):
        x = y * (self.sqrtRin * self.sqrtPt) / (self.eta * self.Vt)
        s = torch.special.i0(x) * self.K * self.eK
        w = lambertw_torch(s)
        return self.eta * self.Vt * (w - self.K)

# =========================================================
# Hybrid Encoder (IDENTICAL to training)
# =========================================================
class HybridEncoderURS(nn.Module):
    def __init__(self, M: int, n: int = 2, L: int = 3):
        super().__init__()
        self.M = M
        self.n = n
        self.k = int(math.log2(M))
        self.L = L

        self.amp_table = self.build_amp_table(M, n, self.k)

        self.theta_re = nn.Parameter(torch.randn(L, n))
        self.theta_im = nn.Parameter(torch.randn(L, n))

        self.dev_re = qml.device("lightning.qubit", wires=n)
        self.dev_im = qml.device("lightning.qubit", wires=n)

        @qml.qnode(self.dev_re, interface="torch")
        def qnode_re(ampvec, weights):
            qml.AmplitudeEmbedding(ampvec, wires=range(n), normalize=True)
            for l in range(L):
                self.pqc_layer(weights[l])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(self.dev_im, interface="torch")
        def qnode_im(ampvec, weights):
            qml.AmplitudeEmbedding(ampvec, wires=range(n), normalize=True)
            for l in range(L):
                self.pqc_layer(weights[l])
            return qml.expval(qml.PauliZ(0))

        self.qnode_re = qnode_re
        self.qnode_im = qnode_im

    @staticmethod
    def build_amp_table(M, n, k):
        table = torch.zeros(M, 2**n)
        shift = max(n - k, 0)
        for j in range(M):
            table[j, j << shift] = 1.0
        return table

    def pqc_layer(self, weights):
        for i in range(self.n):
            qml.RY(math.pi * weights[i], wires=i)
        for i in range(self.n - 1):
            qml.CNOT(wires=[i, i + 1])

    def forward(self, symbols):
        uniq, inv = torch.unique(symbols, sorted=True, return_inverse=True)
        amp_uniq = self.amp_table[uniq].to(symbols.device)

        outs_re, outs_im = [], []
        for u in range(amp_uniq.size(0)):
            amp = amp_uniq[u]
            outs_re.append(self.qnode_re(amp, self.theta_re))
            outs_im.append(self.qnode_im(amp, self.theta_im))

        x_re = torch.stack(outs_re)[inv].unsqueeze(-1)
        x_im = torch.stack(outs_im)[inv].unsqueeze(-1)

        x_mag = torch.sqrt(x_re**2 + x_im**2 + 1e-12)
        power = torch.mean(x_mag**2, dim=0, keepdim=True) + 1e-12
        return x_mag / torch.sqrt(power)

# =========================================================
# Decoder (IDENTICAL)
# =========================================================
class Decoder2(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.fc1 = nn.Linear(1, M)
        self.fc2 = nn.Linear(M, M)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)

# =========================================================
# Autoencoder (IDENTICAL NAMES)
# =========================================================
class Autoencoder2(nn.Module):
    def __init__(self, M, eta, Vt, K, Rin, Pt):
        super().__init__()
        self.enc2  = HybridEncoderURS(M=M, n=2, L=3)
        self.nonli = RectifierEH(eta, Vt, K, Rin, Pt)
        self.dec2  = Decoder2(M)

    def forward(self, symbols):
        x = self.enc2(symbols)
        v = self.nonli(x)
        noisy = v + noise_std * torch.randn_like(v)
        return self.dec2(noisy)

# =========================================================
# Learned evaluation (YOUR WAY)
# =========================================================
@torch.no_grad()
def evaluate_ser_learned(model, Nsym=200000, batch=50000):
    model.eval()
    errors = 0
    for _ in range(Nsym//batch):
        syms = torch.randint(0, M, (batch,), device=device)
        logits = model(syms)
        preds  = logits.argmax(dim=1)
        errors += (preds != syms).sum().item()
    return errors / Nsym

# =========================================================
# Baseline: simulate SER for a given unipolar constellation y_levels (E[y^2]=1)
# Detection: nearest neighbor in rectified voltage domain
# =========================================================
@torch.no_grad()
def evaluate_ser_constellation(y_levels_np, Pt, Nsym=200000, batch=50000):
    y_levels = torch.tensor(y_levels_np, dtype=torch.float32, device=device).view(M, 1)
    rect = RectifierEH(eta, Vt, K, Rin, Pt).to(device)

    # noiseless rectified outputs per symbol (decision points)
    v_levels = rect(y_levels).view(M)

    errors = 0
    loops = Nsym // batch
    for _ in range(loops):
        syms = torch.randint(0, M, (batch,), device=device)
        y = y_levels[syms]                  # (batch,1)
        v = rect(y)                         # (batch,1)
        r = v + noise_std * torch.randn_like(v)

        # nearest neighbor in voltage domain
        # distances: (batch, M)
        d = torch.abs(r - v_levels.view(1, M))
        preds = torch.argmin(d, dim=1)
        errors += (preds != syms).sum().item()

    return errors / (loops * batch)

# =========================================================
# Algorithmic baseline:
# Choose voltages v_m = m*Delta and invert rectifier to get y_m,
# then tune Delta so that mean(y_m^2)=1  (same power norm as learned encoder)
# =========================================================
def rectifier_forward_scalar(y, Pt):
    # y >= 0, scalar float
    sqrtRin = math.sqrt(Rin)
    sqrtPt  = math.sqrt(2 * Pt)
    x = y * (sqrtRin * sqrtPt) / (eta * Vt)
    s = i0(x) * K * math.exp(K)
    w = lambertw(s).real
    return eta * Vt * (w - K)

def invert_rectifier_for_voltage(v_target, Pt):
    if v_target <= 0.0:
        return 0.0

    def f(y):
        return rectifier_forward_scalar(y, Pt) - v_target

    # bracket: increase hi until f(hi) > 0
    lo = 0.0
    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket root for inversion (hi exploded).")

    sol = root_scalar(f, bracket=[lo, hi], method="bisect", xtol=1e-10, rtol=1e-10, maxiter=200)
    return float(sol.root)

def design_algorithmic_levels(Pt, M=4):
    # Bisection on Delta to satisfy mean(y^2)=1
    def levels_for_delta(Delta):
        y = np.zeros(M, dtype=np.float64)
        for m in range(1, M):
            y[m] = invert_rectifier_for_voltage(m * Delta, Pt)
        return y

    # find a Delta_high where mean(y^2) > 1
    Delta_lo = 0.0
    Delta_hi = 1e-6
    for _ in range(80):
        y_hi = levels_for_delta(Delta_hi)
        ms_hi = np.mean(y_hi**2)
        if ms_hi > 1.0:
            break
        Delta_hi *= 2.0
    else:
        raise RuntimeError("Could not find Delta_hi making mean(y^2) > 1")

    # bisection
    for _ in range(70):
        Delta_mid = 0.5 * (Delta_lo + Delta_hi)
        y_mid = levels_for_delta(Delta_mid)
        ms = np.mean(y_mid**2)
        if ms > 1.0:
            Delta_hi = Delta_mid
        else:
            Delta_lo = Delta_mid

    y = levels_for_delta(Delta_hi)
    # final tiny normalize safety (should be ~1 already)
    ms = np.mean(y**2)
    y = y / math.sqrt(ms + 1e-15)
    return y.astype(np.float32)

# =========================================================
# Unipolar M-PAM levels: [0,1,2,3] normalized to E[y^2]=1
# =========================================================
def unipolar_mpam_levels(M=4):
    y = np.arange(M, dtype=np.float64)  # 0..M-1
    ms = np.mean(y**2)
    y = y / math.sqrt(ms + 1e-15)
    return y.astype(np.float32)

# =========================================================
# Theoretical M-PAM SER on linear AWGN (reference "M=4")
# Using SNR = Es/N0 (here we use Pt/noise^2 same as your SNR axis)
# Ps = 2*(M-1)/M * Q( sqrt(6/(M^2-1) * SNR) )
# =========================================================
def Q(x):
    return 0.5 * erfc(x / math.sqrt(2))

def ser_mpam_awgn(M, snr_lin):
    return 2.0 * (M - 1) / M * Q(math.sqrt(6.0 / (M**2 - 1) * snr_lin))

# =========================================================
# MAIN
# =========================================================
Pt_dBm_list = [10, 5, 0, -5, -10, -15, -20]
Pt_list = [10**((dBm - 30) / 10) for dBm in Pt_dBm_list]

ser_learned   = []
ser_algo      = []
ser_unipolar  = []
ser_m4_awgn   = []

# fixed baseline unipolar levels (power-normalized)
y_uni = unipolar_mpam_levels(M)

for dBm, Pt in zip(Pt_dBm_list, Pt_list):
    print(f"\nPt = {dBm} dBm")

    # ---------------- Learned (Quantum) ----------------
    model = Autoencoder2(M, eta, Vt, K, Rin, Pt).to(device)
    ckpt = torch.load(f"autoenc2_4PAM_Pt{dBm}dBm.pth", map_location=device)

    # support both: raw state_dict OR {"state_dict": ...}
    if isinstance(ckpt, dict) and any(k.startswith("enc2") or k.startswith("dec2") for k in ckpt.keys()):
        state = ckpt
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    ser_learned.append(evaluate_ser_learned(model))

    # ---------------- Algorithmic baseline ----------------
    y_alg = design_algorithmic_levels(Pt, M=M)
    ser_algo.append(evaluate_ser_constellation(y_alg, Pt))

    # ---------------- Unipolar M-PAM baseline ----------------
    ser_unipolar.append(evaluate_ser_constellation(y_uni, Pt))

    # ---------------- M=4 (linear AWGN theory reference) ----------------
    snr_lin = Pt / (noise_std**2)
    ser_m4_awgn.append(ser_mpam_awgn(M, snr_lin))

# =========================================================
# Plot (ALL CURVES)
# =========================================================
snr_db = 10 * np.log10(np.array(Pt_list) / (noise_std**2))

plt.figure(figsize=(7, 5))

plt.semilogy(snr_db, ser_algo,     'x-', label="Algorithmic")
plt.semilogy(snr_db, ser_learned,  'o-', label="Learned")
plt.semilogy(snr_db, ser_unipolar, '^-', label="Unipolar M-PAM")
plt.semilogy(snr_db, ser_m4_awgn,  '*-', label="M=4")

plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()