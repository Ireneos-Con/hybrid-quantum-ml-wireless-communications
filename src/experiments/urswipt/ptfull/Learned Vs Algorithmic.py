import os
import numpy as np
from scipy.special import i0, i1, lambertw
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from mpmath import erfc, sqrt

# =========================================================
# helper functions
# =========================================================

def i0_inv(target, tol=1e-9, maxiter=500):
    if target < 1.0:
        raise ValueError("I0(z) >= 1 required")
    if abs(target - 1.0) < tol:
        return 0.0
    z = np.arccosh(target)
    for _ in range(maxiter):
        f  = i0(z) - target
        df = i1(z)
        if abs(df) < 1e-12:
            break
        z_new = z - f / df
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    a, b = 0.0, max(2*z, 1.0)
    for _ in range(60):
        m = 0.5*(a+b)
        if abs(i0(m) - target) < tol:
            return m
        if (i0(a)-target)*(i0(m)-target) < 0:
            b = m
        else:
            a = m
    return 0.5*(a+b)

def inverse_rectifier(v, eta, Vt, Rin, K, Pt):
    v = np.atleast_1d(v)
    arg = (1 + v/(K*eta*Vt)) * np.exp(v/(eta*Vt))
    z   = np.array([i0_inv(a) for a in arg])
    return (eta*Vt)/(np.sqrt(Rin)*np.sqrt(2*Pt)) * z

def find_dmin(M, eta, Vt, Rin, K, Pt, tol=1e-6):
    def g(d):
        vbar = np.arange(M)*d
        y    = inverse_rectifier(vbar, eta, Vt, Rin, K, Pt)
        return np.mean(y**2) - 1.0
    Δ0 = np.sqrt(6/((M-1)*(2*M-1)))
    sol = root_scalar(g, bracket=[1e-6, M*Δ0], method='bisect', xtol=tol)
    return sol.root

def steady_state_voltage(x, Pt, eta, Vt, Rin, K):
    z   = x * np.sqrt(Rin * 2 * Pt) / (eta * Vt)
    arg = i0(z) * K * np.exp(K)
    W   = lambertw(arg)
    return eta * Vt * (W.real - K)

# =========================================================
# LOAD LEARNED CONSTELLATION (NEW VERSION)
# =========================================================

def load_learned_constellation(Pt_dBm):
    """
    Learned constellations exist for:
    10, 5, 0, -5, -10, -15, -20 dBm
    """

    # convert e.g. -5 -> "-5", 10 -> "10"
    if float(Pt_dBm).is_integer():
        s = str(int(Pt_dBm))
    else:
        s = str(Pt_dBm).replace('.', 'p')

    fname = f"urqswipt_imatrix_Pt{s}dBm.npz"

    if not os.path.exists(fname):
        return None

    return np.load(fname)["x"]
# =========================================================
# system constants
# =========================================================

M = 4
eta, Vt, Rin = 1.05, 25.85e-3, 50
Is, R        = 5e-6, 8.25e3
K            = (Is * R) / (eta * Vt)

Pt_dBm_lst = sorted([
    10, 7.5, 5, 2.5, 0,
    -2.5, -5, -7.5, -10,
    -12.5, -15, -17.5, -20, -22.5
])

# =========================================================
# FIGURE 1: Constellation vs Pt
# =========================================================

plt.figure(figsize=(6,4))

for Pt_dBm in Pt_dBm_lst:
    Pt = 10**((Pt_dBm - 30)/10)

    # --- Analytic constellation ---
    d_min = find_dmin(M, eta, Vt, Rin, K, Pt)
    vbar  = np.arange(M)*d_min
    y_a   = inverse_rectifier(vbar, eta, Vt, Rin, K, Pt)

    plt.scatter(
        y_a,
        [Pt_dBm]*M,
        s=45,
        edgecolor='k',
        alpha=0.85,
        label="Analytic" if Pt_dBm == Pt_dBm_lst[0] else None
    )

    # --- Learned constellation (ONLY if exists) ---
    y_l = load_learned_constellation(Pt_dBm)
    if y_l is not None:
        plt.scatter(
            y_l,
            [Pt_dBm]*M,
            s=70,
            marker="x",
            color="red",
            label="Learned" if Pt_dBm == -2.5 else None
        )

plt.xlabel("Amplitude")
plt.ylabel("Transmit power $P_t$ (dBm)")
plt.title("4-PAM UR-SWIPT Constellations\nAnalytic vs Learned")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# FIGURE 2: SER vs SNR
# =========================================================

noise_std = 2e-2
Nsym      = 200_000
batch     = 20_000

snr_db_all   = []
ser_analytic = []

snr_db_l     = []
ser_learned  = []

for Pt_dBm in Pt_dBm_lst:
    Pt = 10**((Pt_dBm - 30)/10)
    snr_db = 10*np.log10(Pt / noise_std**2)

    # --- Analytic constellation ---
    d_min = find_dmin(M, eta, Vt, Rin, K, Pt)
    vbar  = np.arange(M)*d_min
    y     = inverse_rectifier(vbar, eta, Vt, Rin, K, Pt)
    y_a   = steady_state_voltage(y, Pt, eta, Vt, Rin, K)

    # Analytic SER
    err_a = 0
    for _ in range(Nsym // batch):
        s = np.random.randint(0, M, batch)
        r = y_a[s] + noise_std*np.random.randn(batch)
        dec = np.abs(r[:,None] - y_a[None,:]).argmin(axis=1)
        err_a += np.count_nonzero(dec != s)

    ser_analytic.append(err_a / Nsym)
    snr_db_all.append(snr_db)

    # Learned SER (ONLY if learned exists)
    y_l = load_learned_constellation(Pt_dBm)
    if y_l is not None:
        err_l = 0
        for _ in range(Nsym // batch):
            s = np.random.randint(0, M, batch)
            r = y_l[s] + noise_std*np.random.randn(batch)
            dec = np.abs(r[:,None] - y_l[None,:]).argmin(axis=1)
            err_l += np.count_nonzero(dec != s)

        ser_learned.append(err_l / Nsym)
        snr_db_l.append(snr_db)

# --- Plot SER ---
plt.figure(figsize=(7,5))
plt.semilogy(snr_db_all, ser_analytic, 'o-', label="4-PAM Analytic")
plt.semilogy(snr_db_l, ser_learned,  'x--', label="4-PAM Learned")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.title("SER vs SNR — 4-PAM UR-SWIPT")
plt.legend()
plt.ylim(1e-4, 1)
plt.tight_layout()
plt.show()
