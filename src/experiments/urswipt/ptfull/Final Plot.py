import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import savemat
from scipy.special import i0, i1, lambertw
from scipy.optimize import root_scalar

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— helper functions ———

def i0_inv(target, tol=1e-16, maxiter=500):
    """Invert I0(z)=target for z>=0 via Newton–Raphson + bisection."""
    if target < 1.0:
        raise ValueError(f"I0(z)>=1; got target={target:.6f}<1")
    if abs(target - 1.0) < tol:
        return 0.0
    # initial guess
    z = np.arccosh(target)
    for _ in range(maxiter):
        f  = i0(z) - target
        df = i1(z)
        if abs(df) < 1e-12:
            break
        z_new = z - f/df
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    # fallback bisection
    a, b = 0.0, max(2*z,1.0)
    fa, fb = i0(a) - target, i0(b) - target
    for _ in range(50):
        m = 0.5*(a+b)
        fm = i0(m) - target
        if abs(fm) < tol:
            return m
        if fa*fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

def inverse_rectifier(v, eta, Vt, Rin, K, Pt):
    """
    y = f_nl^{-1}(v):
      v = harvested DC voltage,
      y = RF amplitude before normalization.
    """
    v = np.atleast_1d(v)
    # solve I0(z) = (1 + v/(K η Vt)) * exp(v/(η Vt))
    arg = (1 + v/(K*eta*Vt)) * np.exp(v/(eta*Vt))
    z   = np.array([i0_inv(a) for a in arg])
    # recover RF envelope
    return (eta*Vt)/(np.sqrt(Rin)*np.sqrt(2*Pt)) * z

def find_dmin(M, eta, Vt, Rin, K, Pt, tol=1e-6):
    """Solve ½∑[f^{-1}(i d)]² = 1 for d>0 by bisection."""
    def g(d):
        vbar = np.arange(M)*d
        y    = inverse_rectifier(vbar, eta, Vt, Rin, K, Pt)
        return np.mean(y**2) - 1.0
    # initial bracket from standard unipolar‐PAM Δ₀
    Δ0 = np.sqrt(6/((M-1)*(2*M-1)))
    sol = root_scalar(g,
                      bracket=[1e-6, M*Δ0],
                      method='bisect',
                      xtol=tol)
    return sol.root


def steady_state_voltage(x, Pt):
    K = (Is * R) / (eta * Vt)
    z = x * np.sqrt(Rin * 2 * Pt) / (eta * Vt)
    arg = i0(z) * K * np.exp(K)
    W = lambertw(arg)
    vL = eta * Vt * (W.real - K)
    return vL

# system parameters
R, eta, Vt = 8.25e3, 1.05, 25.85e-3
Is, Rin    = 5e-6, 50
K          = (Is * R) / (eta * Vt)
noise_std  = 2e-2   # fixed σ

# Lambert W for rectifier
def lambertw_torch(x, iters=20):
    w = torch.log1p(x)
    for _ in range(iters):
        ew = torch.exp(w)
        f  = w*ew - x
        dw = ew*(w+1) - (w+2)*f/(2*(w+1)+1e-8)
        w = w - f/dw
    return w

# shared, pretrained AE1 encoder
class Encoder1(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.fc1a = nn.Linear(M, M)
        self.fc2a = nn.Linear(M, 1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1a(x))
        x = self.fc2a(x)
        return torch.abs(x / (x.pow(2).mean().sqrt() + 1e-8))

# rectifier NL block
class NonLinear(nn.Module):
    def __init__(self, Pt):
        super().__init__()
        self.eta, self.Vt, self.K, self.Rin = eta, Vt, K, Rin
        self.sqrtPt = math.sqrt(2*Pt)
    def forward(self, y):
        x = y * (math.sqrt(self.Rin)*self.sqrtPt)/(self.eta*self.Vt)
        s = torch.special.i0(x)*self.K*math.exp(self.K)
        w = lambertw_torch(s)
        return self.eta*self.Vt*(w - self.K)

# AWGN
class AWGNChannel(nn.Module):
    def __init__(self, _):  # noise_std fixed globally
        super().__init__()
    def forward(self, x):
        return x + torch.randn_like(x)*noise_std

# trainable decoder3
class Decoder3(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.fc1 = nn.Linear(1, M)
        self.fc2 = nn.Linear(M, M)
    def forward(self, x):
        return self.fc2(F.leaky_relu(self.fc1(x)))

# full AE3
class Autoencoder3(nn.Module):
    def __init__(self, M, Pt):
        super().__init__()
        self.enc3  = encoder1     # shared & frozen
        self.nonli = NonLinear(Pt)
        self.chan3 = AWGNChannel(None)
        self.dec3  = Decoder3(M)
    def forward(self, x):
        x = self.enc3(x)
        x = self.nonli(x)
        x = self.chan3(x)
        return self.dec3(x)

# load shared encoder1 (4-PAM)
encoder1 = Encoder1(4).to(device)
encoder1.load_state_dict(torch.load("encoder1_4PAM.pth", map_location=device))
encoder1.eval()
for p in encoder1.parameters(): p.requires_grad = False

# evaluation fn
def evaluate_ser_ae3(ae3, M, num_symbols=2_000_000, batch_size=200_000):
    ae3.eval()
    errors = 0
    total  = 0
    with torch.no_grad():
        for _ in range(num_symbols//batch_size):
            syms   = torch.randint(0, M, (batch_size,), device=device)
            oh     = F.one_hot(syms, M).float().to(device)
            logits = ae3(oh)
            preds  = logits.argmax(dim=1)
            errors += (preds != syms).sum().item()
            total  += batch_size
    return errors/total

# Pt grid
#Pt_dBm_list = sorted([10,7.5, 5,2.5, 0,-2.5, -5,-7.5, -10,-12.5, -15,-17.5, -20,-22.5])

Pt_dBm_list = sorted([10,7.5, 5,2.5, 0,-2.5, -5,-7.5, -10,-12.5, -15,-17.5, -20])

# store results
ser_4 = []
for dBm in Pt_dBm_list:
    Pt_lin = 10**((dBm-30)/10)
    ae3 = Autoencoder3(4, Pt_lin).to(device)
    ae3.dec3.load_state_dict(
        torch.load(f"decoder3_fixedEnc_4PAM_Pt{dBm}dBm.pth", map_location=device)
    )
    ser_4.append(evaluate_ser_ae3(ae3, 4))

# now 8-PAM: re-load encoder1 for 8-PAM
encoder1_8 = Encoder1(8).to(device)
encoder1_8.load_state_dict(torch.load("encoder1_8PAM.pth", map_location=device))
encoder1_8.eval()
for p in encoder1_8.parameters(): p.requires_grad = False

ser_8 = []
for dBm in Pt_dBm_list:
    Pt_lin = 10**((dBm-30)/10)
    ae3 = Autoencoder3(8, Pt_lin).to(device)
    ae3.enc3 = encoder1_8
    ae3.dec3.load_state_dict(
        torch.load(f"decoder3_fixedEnc_8PAM_Pt{dBm}dBm.pth", map_location=device)
    )
    ser_8.append(evaluate_ser_ae3(ae3, 8))






# Rectifier EH nonlinearity
class RectifierEH(nn.Module):
    def __init__(self, eta, Vt, K, Rin, Pt):
        super().__init__()
        self.eta     = eta
        self.Vt      = Vt
        self.K       = K
        self.eK      = torch.tensor(math.exp(K), device=device)
        self.sqrtRin = torch.tensor(math.sqrt(Rin), device=device)
        self.sqrtPt  = torch.tensor(math.sqrt(2*Pt),  device=device)

    def forward(self, y):
        x = (self.sqrtPt * y) * (self.sqrtRin / (self.eta * self.Vt))
        x = lambertw_torch(torch.i0(x) * self.K * self.eK)
        return self.eta * self.Vt * (x - self.K)

# simple M→1 encoder
class Encoder2(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.fc1 = nn.Linear(M, M)
        self.fc2 = nn.Linear(M, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        norm = torch.sqrt(torch.mean(x**2) + 1e-8)
        return torch.abs(x) / norm

# simple 1→M decoder
class Decoder2(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.fc1 = nn.Linear(1, M)
        self.fc2 = nn.Linear(M, M)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)


# Autoencoder: encoder → EH → AWGN(noise_std) → decoder
class Autoencoder2(nn.Module):
    def __init__(self, M, Pt):
        super().__init__()
        self.enc2  = Encoder2(M)
        self.nonli = RectifierEH(eta, Vt, K, Rin, Pt)
        self.dec2  = Decoder2(M)

    def forward(self, x):
        x_enc  = self.enc2(x)                       # encode to scalar
        v      = self.nonli(x_enc)                  # harvested DC voltage
        noisy  = v + torch.randn_like(v) * noise_std
        logits = self.dec2(noisy)                   # back to M‑dim logits
        return logits, v                            # <— return both

# evaluate SER over many symbols
def evaluate_metrics(model, M, num_symbols=2_000_000, batch_size=200_000):
    model.eval()
    errors = total = 0
    P_acc  = 0.0
    with torch.no_grad():
        for _ in range(num_symbols // batch_size):
            syms    = torch.randint(0, M, (batch_size,), device=device)
            oh      = F.one_hot(syms, M).float().to(device)
            logits, v = model(oh)
            preds   = logits.argmax(dim=1)
            errors += (preds != syms).sum().item()
            total  += batch_size
            P_acc  += (v.pow(2) / R).sum().item()
    ser = errors/total
    P_harv = P_acc/total
    return ser, P_harv

# Pt values (dBm), sorted ascending
Pt_dBm_lst = sorted([10,7.5, 5,2.5, 0,-2.5, -5,-7.5, -10,-12.5, -15,-17.5, -20])
Pt_list     = [10**((dBm - 30)/10) for dBm in Pt_dBm_list]

# storage
ser_results = {}
constellations = {}
harvested = {}

for M in (4, 8):
    sers = []
    consts = {}
    hrv=[]

    for PtdBm, Pt in zip(Pt_dBm_list, Pt_list):
        # pick correct filename
        if M == 4:
            fname = f"autoenc2_4PAM_Pt{PtdBm}dBm.pth"
        else:
            fname = f"autoenc2_Pt{PtdBm}dBm.pth"

        # load and eval
        ae = Autoencoder2(M, Pt).to(device)
        ae.load_state_dict(torch.load(fname, map_location=device))
        ae.eval()

        # SER
        ser, P_harv = evaluate_metrics(ae, M)
        sers.append(ser)
        hrv.append(P_harv)
        # constellation amplitudes
        with torch.no_grad():
            eye  = torch.eye(M, device=device)
            amps = ae.enc2(eye).cpu().numpy().flatten()
        consts[PtdBm] = np.sort(amps)

    ser_results[M] = sers
    constellations[M] = consts
    harvested[M] = hrv



# ——— sweep setup ———
Ms         = [4, 8]
Pt_dBm_lst = sorted([10,7.5, 5,2.5, 0,-2.5, -5,-7.5, -10,-12.5, -15,-17.5, -20])
#Pt_dBm_lst = sorted([10,7.5, 5,2.5, 0,-2.5, -5,-7.5, -10,-12.5, -15,-17.5, -20,-22.5])
Nsym       = 2_000_000        # Monte‑Carlo symbols
batch      = 200_000         # per‐batch size
# ——— run SER sweep ———
ser_results1 = {M: [] for M in Ms}

for M in Ms:
    for Pt_dBm in Pt_dBm_lst:
        Pt     = 10**((Pt_dBm-30)/10)      # dBm→linear Watts
        # 1) find unit‑power spacing & constellation
        d_min  = find_dmin(M, eta, Vt, Rin, K, Pt)
        vbar   = np.arange(M)*d_min
        y    = inverse_rectifier(vbar, eta, Vt, Rin, K, Pt)
        y_t = steady_state_voltage(y,Pt)
        # 2) Monte‑Carlo SER
        errors = 0
        for _ in range(Nsym // batch):
            syms = np.random.randint(0, M, size=batch)
            x    = y_t[syms]
            r    = x + noise_std * np.random.randn(batch)
            # ML decode: nearest neighbor in y_t
            # compute absolute distances and argmin
            idx  = np.abs(r[:,None] - y_t[None,:]).argmin(axis=1)
            errors += np.count_nonzero(idx != syms)
        ser = errors / Nsym
        ser_results1[M].append(ser)

# compute linear Pt and corresponding SNR
Pt_lin = np.array([10**((dBm - 30)/10) for dBm in Pt_dBm_lst])
snr_lin = Pt_lin / (noise_std**2)
snr_db  = 10 * np.log10(snr_lin)



# --- Plot SER vs Pt/σ for both M=4 and M=8 ---
snr_lin = np.array(Pt_list) / (noise_std**2)
snr_db  = 10 * np.log10(snr_lin)




# --- plot everything without labels ---
plt.figure(figsize=(7,5))
for M in (4, 8):
    c = 'blue' if M == 4 else 'red'

    # Algorithmic (ser_results1): solid line + x marker (bigger size)
    plt.semilogy(
        snr_db, ser_results1[M],
        color=c, linestyle='-', marker='x',
        markersize=10,  # <- increased
        alpha=0.8
    )

    # Learned (ser_results): solid line + o marker
    plt.semilogy(
        snr_db, ser_results[M],
        color=c, linestyle='-', marker='o',
        alpha=0.8
    )

    # Unipolar PAM baseline: dashed line + * marker (bigger size)
    y_base = ser_4 if M == 4 else ser_8
    plt.semilogy(
        snr_db, y_base,
        color=c, linestyle='--', marker='*',
        markersize=12,  # <- increased
        alpha=0.8
    )

plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.ylim(1e-4, 1)

# --- custom legend with matching marker sizes ---
legend_elements = [
    Line2D([0], [0],
           color='black', linestyle='-', marker='x',
           markersize=8,  # same as plot
           label='Algorithmic'),
    Line2D([0], [0],
           color='black', linestyle='-', marker='o',
           label='Learned'),
    Line2D([0], [0],
           color='black', linestyle='--', marker='*',
           markersize=12,  # same as plot
           label='Unipolar M-PAM'),
    Line2D([0], [0],
           color='blue', lw=2,
           label='M=4'),
    Line2D([0], [0],
           color='red',  lw=2,
           label='M=8'),
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()


# sweep parameters
Ms         = [4, 8]
Pt_dBm_lst = sorted([10, 5, 0,-5, -10, -15, -20])
#Pt_dBm_lst = sorted([10,7.5, 5,2.5, 0,-2.5, -5,-7.5, -10,-12.5, -15,-17.5, -20,-22.5])


fig, axes = plt.subplots(len(Pt_dBm_lst), len(Ms), figsize=(8, 5), sharex=True, sharey=False)

for col, M in enumerate(Ms):
    for row, Pt_dBm in enumerate(Pt_dBm_lst):
        ax = axes[row, col]

        # 1) learned constellations
        amps = constellations[M][Pt_dBm]
        ax.scatter(
            amps,
            np.full_like(amps, Pt_dBm),
            s=80,
            marker='o',
            color='red',
            label='Learned' if (row==0 and col==0) else "",
        )

        # 2) algorithmic spacing
        Pt   = 10**((Pt_dBm - 30)/10)
        dmin = find_dmin(M, eta, Vt, Rin, K, Pt)
        vbar = np.arange(M) * dmin
        y_t  = inverse_rectifier(vbar, eta, Vt, Rin, K, Pt)
        ax.scatter(
            y_t,
            np.full_like(y_t, Pt_dBm),
            s=80,
            marker='x',
            color='blue',
            label='Algorithmic' if (row==0 and col==0) else "",
        )

        # Titles only for top row
        if row == 0:
            ax.set_title(rf"$M={M}$", fontsize=14)

        # Row labels only in first column
        if col == 0:
            ax.set_ylabel(rf"${Pt_dBm}$ dBm", fontsize=12)

        # Center constellation vertically
        ax.set_ylim(Pt_dBm - 1, Pt_dBm + 1)
        ax.set_yticks([])
        ax.grid(True, axis="x", ls="--", alpha=0.3)  # grid only along x
        ax.tick_params(axis='both', labelsize=10)


# Add one legend for whole figure
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=11, frameon=True, framealpha=0.8)

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.show()


# --- Plot harvested power vs Pt for both M=4 and M=8 ---
plt.figure(figsize=(6,4))
for M, style in [(4, 'o-'), (8, 's-')]:
    plt.plot(Pt_dBm_list, harvested[M], style, lw=2, label=f"{M}-PAM")
plt.xlabel("Transmit Power $P_T$ (dBm)")
plt.ylabel("Average Harvested Power $\\mathcal{E}_h$ (W)")
plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.show()