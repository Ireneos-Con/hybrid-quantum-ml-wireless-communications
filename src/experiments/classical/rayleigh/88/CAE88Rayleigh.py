import sys
assert sys.version_info >= (3, 8)

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
k = 8
M = 2**k
n = 8

batch_size = 5_000
steps_per_epoch = 1000
epochs = 80
EBNO_TRAIN_DB = 10.0
runs = 8

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

def avg_runs(test_fn, runs, seed_base=1000, **kwargs):
    all_bler = []

    for r in range(runs):
        torch.manual_seed(seed_base + r)
        np.random.seed(seed_base + r)

        bler = test_fn(**kwargs)
        all_bler.append(bler)

    return np.mean(all_bler, axis=0)

# =========================================================
# Encoder: M → 2n (TABLE-I COMPLIANT)
# =========================================================
class CAEEncoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.M = M
        self.n = n
        self.fc1 = nn.Linear(M, M)
        self.fc2 = nn.Linear(M, 2 * n)

    def forward(self, symbols):
        B = symbols.size(0)
        x = torch.zeros(B, self.M, device=symbols.device)
        x.scatter_(1, symbols.view(-1, 1), 1.0)

        h = F.relu(self.fc1(x))
        z = self.fc2(h)  # [B, 2n]

        power = (z**2).sum(dim=1, keepdim=True) + 1e-12
        z = z * math.sqrt(self.n) / torch.sqrt(power)
        return z

# =========================================================
# Rayleigh block fading + AWGN + perfect EQ (complex)
# =========================================================
class RayleighEQ(nn.Module):
    def __init__(self, k, n, ebno_db):
        super().__init__()
        self.k = k
        self.n = n
        self.ebno_db = ebno_db

    def forward(self, z):
        B = z.size(0)
        sigma = ebno_db2sigma(self.ebno_db, self.n, self.k)

        xr = z[:, :self.n]
        xi = z[:, self.n:]

        hr = torch.randn(B, 1, device=z.device) / math.sqrt(2)
        hi = torch.randn(B, 1, device=z.device) / math.sqrt(2)

        yr = hr * xr - hi * xi + sigma * torch.randn_like(xr)
        yi = hr * xi + hi * xr + sigma * torch.randn_like(xi)

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
        self.fc1 = nn.Linear(2 * n, M)
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
        self.chan = RayleighEQ(k, n, ebno_db)
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
# BLER vs SNR — CAE
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

# =========================================================
# BLER vs SNR — BPSK Rayleigh (complex, fair with AE)
# =========================================================
@torch.no_grad()
def test_bpsk_rayleigh_eq(snr_dBs, Ntest, k, n, device):
    bler = []

    for ebdb in snr_dBs:
        sigma = ebno_db2sigma(ebdb, n, k)

        # bits
        bits = torch.randint(0, 2, (Ntest, k), device=device)

        # BPSK: {0,1} -> {+1,-1}
        tx = 1.0 - 2.0 * bits.float()

        # zero-pad to n channel uses if needed
        if k < n:
            tx = F.pad(tx, (0, n - k))

        xr = tx
        xi = torch.zeros_like(tx)

        # complex Rayleigh fading
        hr = torch.randn(Ntest, 1, device=device) / math.sqrt(2)
        hi = torch.randn(Ntest, 1, device=device) / math.sqrt(2)

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
Ntest = 200_000
snr_dBs = np.linspace(0, 20, 21)
syms_np = np.random.randint(0, M, size=Ntest)

# ---- CAE BLER ----
bler_cae = avg_runs(
    test_fn=test_bler_cae,
    runs=runs,
    autoenc=model,
    syms_np=syms_np,
    snr_dBs=snr_dBs
)

# ---- BPSK baseline ----
bler_bpsk = avg_runs(
    test_fn=test_bpsk_rayleigh_eq,
    runs=runs,
    snr_dBs=snr_dBs,
    Ntest=Ntest,
    k=k,
    n=n,
    device=device
)

np.savez(
    f"CAE_{n}{k}_vs_BPSK_Rayleigh_2.npz",
    snr_dBs=snr_dBs.astype(np.float32),
    bler_cae_rayleigh=bler_cae.astype(np.float32),
    bler_bpsk_rayleigh=bler_bpsk.astype(np.float32),
    k=np.int32(k),
    n=np.int32(n),
    ebno_train_db=np.float32(EBNO_TRAIN_DB)
)

print(f"Saved: CAE_{n}{k}_vs_BPSK_Rayleigh.npz")