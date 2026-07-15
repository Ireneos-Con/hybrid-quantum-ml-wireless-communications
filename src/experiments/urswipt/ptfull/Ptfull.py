import sys
assert sys.version_info >= (3,5)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.special
import pennylane as qml

# =========================================================
# reproducibility & device
# =========================================================
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# system parameters
# =========================================================
k   = 2
M   = 2**k
R   = 8.25e3
eta = 1.05
Vt  = 25.85e-3
Is  = 5e-6
Rin = 50
K   = (Is * R) / (eta * Vt)

# =========================================================
# training hyperparameters
# =========================================================
batch_size      = 60_000
steps_per_epoch = 1000
epochs          = 60
learning_rate   = 5e-3

# =========================================================
# Eb/N0 → noise std (TORCH SAFE)
# =========================================================
def ebno_db2sigma(ebno_db, device):
    ebno = 10.0 ** (ebno_db / 10.0)
    return torch.tensor(1.0 / math.sqrt(2 * k * ebno), device=device)

# =========================================================
# fast Lambert W (torch)
# =========================================================
def lambertw_torch(x, iters=8):
    w = torch.log1p(x)
    for _ in range(iters):
        ew = torch.exp(w)
        f  = w * ew - x
        w  = w - f / (ew * (w + 1) + 1e-8)
    return w

# =========================================================
# Rectifier EH
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
# AWGN Channel
# =========================================================
class AWGNChannel2(nn.Module):
    def __init__(self, ebno_db):
        super().__init__()
        self.ebno_db = ebno_db

    def forward(self, x):
        sigma = ebno_db2sigma(self.ebno_db, x.device)
        return x + sigma * torch.randn_like(x)

# =========================================================
# Hybrid Quantum Encoder (REAL+IMAG → MAGNITUDE)
# =========================================================
class HybridEncoderURS(nn.Module):
    def __init__(self, M: int, n: int = 4, L: int = 4, init_scale: float = 0.1):
        super().__init__()
        self.M = M
        self.n = n
        self.k = int(math.log2(M))
        self.L = L

        self.amp_table = self.build_amp_table(M, n, self.k)

        self.theta_re = nn.Parameter(torch.randn(L, n) * init_scale)
        self.theta_im = nn.Parameter(torch.randn(L, n) * init_scale)

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

        # batch-wise normalization (stable)
        power = torch.mean(x_mag**2, dim=0, keepdim=True) + 1e-12
        return x_mag / torch.sqrt(power)

# =========================================================
# Decoder
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
# Autoencoder
# =========================================================
class Autoencoder2(nn.Module):
    def __init__(self, M, eta, Vt, K, Rin, Pt, ebno_db_train):
        super().__init__()
        self.enc2  = HybridEncoderURS(M=M, n=4, L=4)
        self.nonli = RectifierEH(eta, Vt, K, Rin, Pt)
        self.chan2 = AWGNChannel2(ebno_db_train)
        self.dec2  = Decoder2(M)

    def forward(self, symbols):
        x = self.enc2(symbols)
        vbar = self.nonli(x)
        noisy = self.chan2(vbar)
        return self.dec2(noisy), vbar

# =========================================================
# Training
# =========================================================
train_ebno_for_Pt = {10:18, 5:23,  0:30, -5:37, -10:44, -15:51, -20:62}
criterion = nn.CrossEntropyLoss()

for PtdBm, ebno_train in train_ebno_for_Pt.items():
    Pt_lin = 10**((PtdBm - 30) / 10)

    ae2 = Autoencoder2(M, eta, Vt, K, Rin, Pt_lin, ebno_train).to(device)
    optimizer = optim.NAdam(ae2.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')

    for epoch in range(epochs):
        ae2.train()
        loss_acc = 0.0

        for _ in range(steps_per_epoch):
            symbols = torch.randint(0, M, (batch_size,), device=device)
            logits, _ = ae2(symbols)
            loss = criterion(logits, symbols)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_acc += loss.item()

        loss_acc /= steps_per_epoch
        scheduler.step(loss_acc)
        print(f"[Pt={PtdBm} dBm] Epoch {epoch+1}/{epochs}  loss={loss_acc:.3e}")

    torch.save(ae2.state_dict(), f"autoenc2_4PAM_Pt{PtdBm}dBm.pth")
    print(f"Saved model for Pt={PtdBm} dBm")
