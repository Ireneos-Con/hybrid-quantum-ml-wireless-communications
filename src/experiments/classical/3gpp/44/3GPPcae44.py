import sys
assert sys.version_info >= (3,8)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# =========================================================
# Reproducibility & device
# =========================================================
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# Parameters
# =========================================================
k = 4
M = 2**k
n = 4

batch_size = 5_000
steps_per_epoch = 1000
epochs = 80
EBNO_TRAIN_DB = 10.0

# =========================================================
# Utilities
# =========================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def ebno_db2sigma(ebno_db, n, k):
    ebno_lin = 10.0**(ebno_db / 10.0)
    return math.sqrt(n / (2.0 * k * ebno_lin))

# =========================================================
# Encoder: M → 2n (TABLE-I COMPLIANT)
# =========================================================
class CAEEncoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.M = M
        self.n = n
        self.fc1 = nn.Linear(M, M)
        self.fc2 = nn.Linear(M, 2*n)

    def forward(self, symbols):
        B = symbols.size(0)
        x = torch.zeros(B, self.M, device=symbols.device)
        x.scatter_(1, symbols.view(-1,1), 1.0)

        h = F.relu(self.fc1(x))
        z = self.fc2(h)  # [B, 2n]

        power = (z**2).sum(dim=1, keepdim=True) + 1e-12
        z = z * math.sqrt(self.n) / torch.sqrt(power)
        return z


#3GPP
class SCM3GPP_UMa_EQ(nn.Module):
    def __init__(self, k, n, ebno_db,
                 velocity_kmh=30.0,
                 fc_hz=1.9e9,
                 Ts=1e-3):
        super().__init__()
        self.k = k
        self.n = n
        self.ebno_db = ebno_db

        # SCM params (TR 25.996 Table 5.1)
        self.N = 6   # paths
        self.M = 20  # subpaths per path

        self.velocity = velocity_kmh / 3.6
        self.fc = fc_hz
        self.Ts = Ts

        self.c = 3e8
        self.lambda_c = self.c / self.fc
        self.fD = self.velocity / self.lambda_c

    def forward(self, z):
        B = z.size(0)
        sigma = ebno_db2sigma(self.ebno_db, self.n, self.k)

        xr = z[:, :self.n]
        xi = z[:, self.n:]

        # --- Generate SCM fading ---
        hr = torch.zeros(B,1)
        hi = torch.zeros(B,1)

        for n in range(self.N):
            Pn = 1.0/self.N
            for m in range(self.M):
                phi = 2*math.pi*torch.rand(B,1)
                theta = 2*math.pi*torch.rand(B,1)
                doppler = 2*math.pi*self.fD*self.Ts*torch.cos(theta)

                hr += math.sqrt(Pn/self.M)*torch.cos(phi + doppler)
                hi += math.sqrt(Pn/self.M)*torch.sin(phi + doppler)

        # --- Channel ---
        yr = hr * xr - hi * xi + sigma * torch.randn_like(xr)
        yi = hr * xi + hi * xr + sigma * torch.randn_like(xi)

        # --- Perfect EQ ---
        h2 = hr**2 + hi**2 + 1e-12
        xr_eq = (yr * hr + yi * hi) / h2
        xi_eq = (yi * hr - yr * hi) / h2

        return torch.cat([xr_eq, xi_eq], dim=1)

# =========================================================
# Decoder: 2n → M
# =========================================================
class Decoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.fc1 = nn.Linear(2*n, M)
        self.fc2 = nn.Linear(M, M)

    def forward(self, y):
        return self.fc2(F.relu(self.fc1(y)))

# =========================================================
# Autoencoder
# =========================================================
class CAE(nn.Module):
    def __init__(self, M, n, k, ebno_db):
        super().__init__()
        self.enc = CAEEncoder(M, n)
        self.chan = SCM3GPP_UMa_EQ(k, n, ebno_db)
        self.dec = Decoder(M, n)

    def forward(self, symbols):
        z = self.enc(symbols)
        y = self.chan(z)
        return self.dec(y)

# =========================================================
# Instantiate & train
# =========================================================
model = CAE(M, n, k, EBNO_TRAIN_DB).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for ep in range(epochs):
    model.train()
    L, S = 0.0, 0.0

    for _ in range(steps_per_epoch):
        syms = torch.randint(0, M, (batch_size,), device=device)
        logits = model(syms)
        loss = criterion(logits, syms)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        L += loss.item()
        S += (preds != syms).float().mean().item()

    print(f"Epoch {ep+1:03d} | loss={L/steps_per_epoch:.3e} | SER={S/steps_per_epoch:.3e}")

# =========================================================
# BLER vs SNR — CAE (Rayleigh)
# =========================================================
@torch.no_grad()
def test_bler_cae(autoenc, syms_np, snr_dBs):
    autoenc.eval()
    syms = torch.from_numpy(syms_np).long().to(device)
    bler = []

    for ebdb in snr_dBs:
        autoenc.chan.ebno_db = ebdb
        preds = autoenc(syms).argmax(dim=1)
        bler.append((preds != syms).float().mean().item())

    return np.array(bler)



@torch.no_grad()
def test_bpsk_3gpp_eq(snr_dBs, Ntest, k, n, device,
                      velocity_kmh=30.0,
                      fc_hz=1.9e9,
                      Ts=1e-3):
    """
    Uncoded BPSK baseline under 3GPP SCM-inspired SISO channel
    with perfect equalization, matched to AE channel logic.
    """
    bler = []

    c = 3e8
    velocity = velocity_kmh / 3.6
    lambda_c = c / fc_hz
    fD = velocity / lambda_c

    N_paths = 6
    M_subpaths = 20

    for ebdb in snr_dBs:
        sigma = ebno_db2sigma(ebdb, n, k)

        # random bits per message
        bits = torch.randint(0, 2, (Ntest, k), device=device)

        # BPSK mapping: {0,1} -> {+1,-1}
        tx = 1.0 - 2.0 * bits.float()

        # zero-pad if k < n
        if k < n:
            tx = F.pad(tx, (0, n-k))

        # real-axis BPSK
        xr = tx
        xi = torch.zeros_like(tx)

        # 3GPP SCM-like fading coefficient
        hr = torch.zeros(Ntest, 1, device=device)
        hi = torch.zeros(Ntest, 1, device=device)

        for p in range(N_paths):
            Pn = 1.0 / N_paths
            for m in range(M_subpaths):
                phi = 2 * math.pi * torch.rand(Ntest, 1, device=device)
                theta = 2 * math.pi * torch.rand(Ntest, 1, device=device)
                doppler = 2 * math.pi * fD * Ts * torch.cos(theta)

                hr += math.sqrt(Pn / M_subpaths) * torch.cos(phi + doppler)
                hi += math.sqrt(Pn / M_subpaths) * torch.sin(phi + doppler)

        # channel
        yr = hr * xr - hi * xi + sigma * torch.randn_like(xr)
        yi = hr * xi + hi * xr + sigma * torch.randn_like(xi)

        # perfect equalization
        h2 = hr**2 + hi**2 + 1e-12
        xr_eq = (yr * hr + yi * hi) / h2

        # hard decision
        bits_hat = (xr_eq[:, :k] < 0).long()

        # BLER
        block_err = (bits_hat != bits).any(dim=1).float()
        bler.append(block_err.mean().item())

    return np.array(bler)

# =========================================================
# Run evaluation + SAVE
# =========================================================
Ntest = 50_000
snr_dBs = np.linspace(0, 20, 21)
syms_np = np.random.randint(0, M, size=Ntest)

# ---- CAE BLER ----
bler_cae = test_bler_cae(model, syms_np, snr_dBs)

# ---- BPSK baseline ----
bler_bpsk = test_bpsk_3gpp_eq(
    snr_dBs=snr_dBs,
    Ntest=Ntest,
    k=k,
    n=n,
    device=device,
    velocity_kmh=30.0,
    fc_hz=1.9e9,
    Ts=1e-3
)

np.savez(
    f"CAE_{n}{k}_vs_BPSK_3GPP.npz",
    snr_dBs=snr_dBs.astype(np.float32),
    bler_cae_3GPP=bler_cae.astype(np.float32),
    bler_bpsk_3GPP=bler_bpsk.astype(np.float32),
    k=np.int32(k),
    n=np.int32(n),
    ebno_train_db=np.float32(EBNO_TRAIN_DB)
)

print(f"Saved: CAE_{n}{k}_vs_BPSK_3GPP.npz")

