import sys
assert sys.version_info >= (3, 8)

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tr25996_scm_channel import TR25996SCMEQ


np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

k = 8
M = 2**k
n = 8

batch_size = 32
steps_per_epoch = 1000
epochs = 80
EBNO_TRAIN_DB = 10.0
runs = 5


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CAEEncoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.M = M
        self.n = n
        self.fc1 = nn.Linear(M, M)
        self.fc2 = nn.Linear(M, 2 * n)

    def forward(self, symbols):
        batch = symbols.size(0)
        x = torch.zeros(batch, self.M, device=symbols.device)
        x.scatter_(1, symbols.view(-1, 1), 1.0)

        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        power = (z**2).sum(dim=1, keepdim=True) + 1e-12
        return z * math.sqrt(self.n) / torch.sqrt(power)


class Decoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.fc1 = nn.Linear(2 * n, M)
        self.fc2 = nn.Linear(M, M)

    def forward(self, y):
        return self.fc2(F.relu(self.fc1(y)))


def make_tr25996_channel(ebno_db):
    return TR25996SCMEQ(
        k,
        n,
        ebno_db,
        scenario="urban_macro",
        velocity_kmh=30.0,
        fc_hz=1.9e9,
        distance_m=500.0,
        sample_time_s=1e-3,
        normalize_fading=True,
        apply_pathloss=False,
        urban_macro_as_deg=8.0,
    )


class CAE(nn.Module):
    def __init__(self, M, n, ebno_db):
        super().__init__()
        self.enc = CAEEncoder(M, n)
        self.chan = make_tr25996_channel(ebno_db)
        self.dec = Decoder(M, n)

    def forward(self, symbols):
        z = self.enc(symbols)
        y = self.chan(z)
        return self.dec(y)


def avg_runs(test_fn, runs, seed_base=1000, **kwargs):
    all_bler = []
    for run_idx in range(runs):
        torch.manual_seed(seed_base + run_idx)
        np.random.seed(seed_base + run_idx)
        all_bler.append(test_fn(**kwargs))
    return np.mean(all_bler, axis=0)


model = CAE(M, n, EBNO_TRAIN_DB).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for ep in range(epochs):
    model.train()
    loss_sum, ser_sum = 0.0, 0.0

    for _ in range(steps_per_epoch):
        syms = torch.randint(0, M, (batch_size,), device=device)
        logits = model(syms)
        loss = criterion(logits, syms)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        loss_sum += loss.item()
        ser_sum += (preds != syms).float().mean().item()

    print(
        f"Epoch {ep + 1:03d} | "
        f"loss={loss_sum / steps_per_epoch:.3e} | "
        f"SER={ser_sum / steps_per_epoch:.3e}"
    )


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
def test_bpsk_3gpp_tr25996(snr_dBs, Ntest):
    bler = []

    for ebdb in snr_dBs:
        channel = make_tr25996_channel(ebdb).to(device)
        bits = torch.randint(0, 2, (Ntest, k), device=device)
        tx = 1.0 - 2.0 * bits.float()
        z = torch.cat([tx, torch.zeros_like(tx)], dim=1)
        y_eq = channel(z)
        bits_hat = (y_eq[:, :k] < 0).long()
        block_err = (bits_hat != bits).any(dim=1).float()
        bler.append(block_err.mean().item())

    return np.array(bler)


Ntest = 200_000
snr_dBs = np.linspace(0, 20, 21)
syms_np = np.random.randint(0, M, size=Ntest)

bler_cae = avg_runs(
    test_fn=test_bler_cae,
    runs=runs,
    autoenc=model,
    syms_np=syms_np,
    snr_dBs=snr_dBs,
)

bler_bpsk = avg_runs(
    test_fn=test_bpsk_3gpp_tr25996,
    runs=runs,
    snr_dBs=snr_dBs,
    Ntest=Ntest,
)

np.savez(
    "CAE_88_vs_BPSK_3GPP_TR25996.npz",
    snr_dBs=snr_dBs.astype(np.float32),
    bler_cae_3gpp_tr25996=bler_cae.astype(np.float32),
    bler_bpsk_3gpp_tr25996=bler_bpsk.astype(np.float32),
    k=np.int32(k),
    n=np.int32(n),
    ebno_train_db=np.float32(EBNO_TRAIN_DB),
    scenario="urban_macro",
    velocity_kmh=np.float32(30.0),
    fc_hz=np.float32(1.9e9),
    distance_m=np.float32(500.0),
    normalized_fading=np.bool_(True),
)

print("Saved: CAE_88_vs_BPSK_3GPP_TR25996.npz")
