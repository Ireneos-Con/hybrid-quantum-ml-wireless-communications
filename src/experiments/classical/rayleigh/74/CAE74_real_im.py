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
n = 7

batch_size = 5_000
steps_per_epoch = 1000
epochs = 80
EBNO_TRAIN_DB = 10.0

# validation for epoch BLER
Nval = 50_000

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
        z = self.fc2(h)

        power = (z**2).sum(dim=1, keepdim=True) + 1e-12
        z = z * math.sqrt(self.n) / torch.sqrt(power)
        return z

# =========================================================
# Rayleigh block fading + AWGN + perfect EQ
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
# Decoder
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
        self.chan = RayleighEQ(k, n, ebno_db)
        self.dec = Decoder(M, n)

    def forward(self, symbols):
        z = self.enc(symbols)
        y = self.chan(z)
        return self.dec(y)

# =========================================================
# Instantiate
# =========================================================
model = CAE(M, n, k, EBNO_TRAIN_DB).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# fixed validation symbols for epoch BLER
val_syms = torch.randint(0, M, (Nval,), device=device)

bler_epoch = []

# =========================================================
# TRAIN
# =========================================================
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

    # ----- BLER @ epoch -----
    model.eval()
    with torch.no_grad():
        preds = model(val_syms).argmax(dim=1)
        bler_ep = (preds != val_syms).float().mean().item()
        bler_epoch.append(bler_ep)

    print(
        f"Epoch {ep+1:03d} | "
        f"loss={L/steps_per_epoch:.3e} | "
        f"SER={S/steps_per_epoch:.3e} | "
        f"BLER_epoch={bler_ep:.3e}"
    )

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
# Hamming (7,4) baseline
# =========================================================
@torch.no_grad()
def hamming74_codebook():
    G = torch.tensor([
        [1,0,0,0,0,1,1],
        [0,1,0,0,1,0,1],
        [0,0,1,0,1,1,0],
        [0,0,0,1,1,1,1],
    ], dtype=torch.int64, device=device)

    msgs = torch.arange(16, device=device)
    bits = ((msgs[:,None] >> torch.tensor([3,2,1,0], device=device)) & 1)
    C = (bits @ G) & 1
    return 1.0 - 2.0 * C.float()

@torch.no_grad()
def test_hamming74_mld(snr_dBs, Ntest):
    X = hamming74_codebook()
    bler = []

    for ebdb in snr_dBs:
        sigma = ebno_db2sigma(ebdb, n, k)
        syms = torch.randint(0, 16, (Ntest,), device=device)
        tx = X[syms]

        g1 = torch.randn(Ntest, device=device) / math.sqrt(2)
        g2 = torch.randn(Ntest, device=device) / math.sqrt(2)
        h  = torch.sqrt(g1**2 + g2**2)

        y = h[:,None] * tx + sigma * torch.randn_like(tx)
        y_eq = y / (h[:,None] + 1e-12)

        d2 = ((y_eq[:,None,:] - X[None,:,:])**2).sum(dim=2)
        preds = d2.argmin(dim=1)
        bler.append((preds != syms).float().mean().item())

    return np.array(bler)

# =========================================================
# RUN + SAVE
# =========================================================
Ntest = 50_000
snr_dBs = np.linspace(0, 20, 21)
syms_np = np.random.randint(0, M, size=Ntest)

bler_cae = test_bler_cae(model, syms_np, snr_dBs)
bler_h74 = test_hamming74_mld(snr_dBs, Ntest)

np.savez(
    "CAE_vs_Hamming74_Rayleigh.npz",
    snr_dBs=snr_dBs.astype(np.float32),
    bler_cae_rayleigh=bler_cae.astype(np.float32),
    bler_hamming74_rayleigh=bler_h74.astype(np.float32),
    bler_epoch=np.array(bler_epoch, dtype=np.float32),
    k=np.int32(k),
    n=np.int32(n),
    ebno_train_db=np.float32(EBNO_TRAIN_DB)
)

print("Saved: CAE_vs_Hamming74_Rayleigh.npz")
