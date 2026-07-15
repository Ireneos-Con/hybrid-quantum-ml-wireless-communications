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



def avg_runs(test_fn, runs, seed_base=1000, **kwargs):
    all_bler = []

    for r in range(runs):
        torch.manual_seed(seed_base + r)
        np.random.seed(seed_base + r)

        bler = test_fn(**kwargs)
        all_bler.append(bler)

    return np.mean(all_bler, axis=0)

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
def test_hamming74_mld_3gpp(snr_dBs, Ntest):

    X = hamming74_codebook().to(device)
    bler = []

    for ebdb in snr_dBs:

        # 3GPP channel
        channel = SCM3GPP_UMa_EQ(k, n, ebdb).to(device)

        syms = torch.randint(0, 16, (Ntest,), device=device)

        tx = X[syms]

        # --- convert to complex baseband ---
        xr = tx
        xi = torch.zeros_like(tx)

        z = torch.cat([xr, xi], dim=1)

        # --- channel ---
        y_eq = channel(z)

        yr = y_eq[:, :n]

        # --- MLD detection ---
        d2 = ((yr[:,None,:] - X[None,:,:])**2).sum(dim=2)

        preds = d2.argmin(dim=1)

        bler.append((preds != syms).float().mean().item())

    return np.array(bler)
# =========================================================
# Run evaluation + SAVE
# =========================================================
Ntest = 100_000
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
# ---- Hamming BLER ----
bler_hamming = avg_runs(
    test_fn=test_hamming74_mld_3gpp,
    runs=runs,
    snr_dBs=snr_dBs,
    Ntest=Ntest
)

np.savez(
    f"CAE_{n}{k}_vs_Hamming_3GPP.npz",
    snr_dBs=snr_dBs.astype(np.float32),
    bler_cae_3gpp=bler_cae.astype(np.float32),
    bler_hamming_3gpp=bler_hamming.astype(np.float32),
    k=np.int32(k),
    n=np.int32(n),
    ebno_train_db=np.float32(EBNO_TRAIN_DB)
)

print(f"Saved: CAE_{n}{k}_vs_Hamming_3GPP.npz")

