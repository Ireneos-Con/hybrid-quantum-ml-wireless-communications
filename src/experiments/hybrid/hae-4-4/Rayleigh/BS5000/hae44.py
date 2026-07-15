import sys
assert sys.version_info >= (3, 8)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pennylane as qml

# =========================================================
# Reproducibility & device
# =========================================================
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")   # quantum runs on CPU
print("Device:", device)

# =========================================================
# Parameters (match CAE)
# =========================================================
k = 4
M = 2**k
n = 4

batch_size = 5000
steps_per_epoch = 1000
epochs = 80
EBNO_TRAIN_DB = 10.0
Nval = 50000

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

def build_amp_table(M, n, k):
    table = torch.zeros(M, 2**n)
    shift = max(n - k, 0)
    for j in range(M):
        table[j, j << shift] = 1.0
    return table

# =========================================================
# Quantum Encoder  -> returns [B, 2n]
# =========================================================
class HAEEncoder(nn.Module):
    def __init__(self, M, n, L=3):
        super().__init__()
        self.M = M
        self.n = n
        self.k = int(math.log2(M))
        self.L = L

        self.amp_table = build_amp_table(M, n, self.k).to(device)

        self.theta_re = nn.Parameter(torch.randn(L, n) * 0.1)
        self.theta_im = nn.Parameter(torch.randn(L, n) * 0.1)

        self.dev_re = qml.device("lightning.qubit", wires=n)
        self.dev_im = qml.device("lightning.qubit", wires=n)

        @qml.qnode(self.dev_re, interface="torch")
        def qnode_re(ampvec, weights):
            qml.AmplitudeEmbedding(ampvec, wires=range(n), normalize=True)
            for l in range(L):
                for i in range(n):
                    qml.RY(math.pi * weights[l, i], wires=i)
                for i in range(n-1):
                    qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        @qml.qnode(self.dev_im, interface="torch")
        def qnode_im(ampvec, weights):
            qml.AmplitudeEmbedding(ampvec, wires=range(n), normalize=True)
            for l in range(L):
                for i in range(n):
                    qml.RY(math.pi * weights[l, i], wires=i)
                for i in range(n-1):
                    qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        self.qnode_re = qnode_re
        self.qnode_im = qnode_im

    def forward(self, symbols):
        uniq, inv = torch.unique(symbols, sorted=True, return_inverse=True)
        amp_uniq = self.amp_table[uniq]

        outs_re = []
        outs_im = []

        for u in range(amp_uniq.size(0)):
            amp = amp_uniq[u]
            xr = torch.stack(self.qnode_re(amp, self.theta_re))
            xi = torch.stack(self.qnode_im(amp, self.theta_im))
            outs_re.append(xr)
            outs_im.append(xi)

        outs_re = torch.stack(outs_re)
        outs_im = torch.stack(outs_im)

        x_re = outs_re[inv]
        x_im = outs_im[inv]

        power = (x_re**2 + x_im**2).sum(dim=1, keepdim=True) + 1e-12
        scale = math.sqrt(self.n) / torch.sqrt(power)

        x_re = x_re * scale
        x_im = x_im * scale

        return torch.cat([x_re, x_im], dim=1)  # [B, 2n]

# =========================================================
# Rayleigh + AWGN + Perfect EQ (identical to CAE)
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

        hr = torch.randn(B, 1) / math.sqrt(2)
        hi = torch.randn(B, 1) / math.sqrt(2)

        yr = hr * xr - hi * xi + sigma * torch.randn_like(xr)
        yi = hr * xi + hi * xr + sigma * torch.randn_like(xi)

        h2 = hr**2 + hi**2 + 1e-12
        xr_eq = (yr * hr + yi * hi) / h2
        xi_eq = (yi * hr - yr * hi) / h2

        return torch.cat([xr_eq, xi_eq], dim=1)

# =========================================================
# Decoder (same as CAE)
# =========================================================
class Decoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.fc1 = nn.Linear(2*n, M)
        self.fc2 = nn.Linear(M, M)

    def forward(self, y):
        return self.fc2(F.relu(self.fc1(y)))

# =========================================================
# Full Autoencoder
# =========================================================
class HAE(nn.Module):
    def __init__(self, M, n, k, ebno_db):
        super().__init__()
        self.enc = HAEEncoder(M, n)
        self.chan = RayleighEQ(k, n, ebno_db)
        self.dec = Decoder(M, n)

    def forward(self, symbols):
        z = self.enc(symbols)
        y = self.chan(z)
        return self.dec(y)

# =========================================================
# Instantiate
# =========================================================
model = HAE(M, n, k, EBNO_TRAIN_DB).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_hist = []
ser_hist = []
bler_epoch = []

val_syms = torch.randint(0, M, (Nval,))

# =========================================================
# TRAIN
# =========================================================
for ep in range(epochs):
    model.train()
    L, S = 0.0, 0.0

    for _ in range(steps_per_epoch):
        syms = torch.randint(0, M, (batch_size,))
        logits = model(syms)
        loss = criterion(logits, syms)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        L += loss.item()
        S += (preds != syms).float().mean().item()

    L /= steps_per_epoch
    S /= steps_per_epoch

    loss_hist.append(L)
    ser_hist.append(S)

    model.eval()
    with torch.no_grad():
        preds = model(val_syms).argmax(dim=1)
        bler_ep = (preds != val_syms).float().mean().item()
        bler_epoch.append(bler_ep)

    print(f"Epoch {ep+1:03d} | loss={L:.3e} | SER={S:.3e} | BLER_epoch={bler_ep:.3e}")

# =========================================================
# BLER vs SNR
# =========================================================
@torch.no_grad()
def test_bler(model, syms_np, snr_dBs):
    model.eval()
    syms = torch.from_numpy(syms_np).long()
    bler = []

    for ebdb in snr_dBs:
        model.chan.ebno_db = ebdb
        preds = model(syms).argmax(dim=1)
        bler.append((preds != syms).float().mean().item())

    return np.array(bler)

Ntest = 50000
snr_dBs = np.linspace(0, 20, 21)
syms_np = np.random.randint(0, M, size=Ntest)

bler_hae = test_bler(model, syms_np, snr_dBs)

print("SNR:", snr_dBs)
print("HAE BLER:", bler_hae)

# =========================================================
# SAVE (same keys as original HAE script)
# =========================================================
results = {
    "loss": np.array(loss_hist, dtype=np.float32),
    "ser": np.array(ser_hist, dtype=np.float32),
    "snr_dBs": snr_dBs.astype(np.float32),
    "bler_ae_rayleigh": bler_hae.astype(np.float32),
    "bler_epoch": np.array(bler_epoch, dtype=np.float32),
}

np.savez("hae44.npz", **results)
print("Saved: hae44.npz")
