import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# =========================================================
# device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# Hybrid Encoder (IDENTICAL με URSWIPT.py)
# =========================================================
class HybridEncoderURS(nn.Module):
    def __init__(self, M: int, n: int = 2, L: int = 3, init_scale: float = 0.1):
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

        power = torch.mean(x_mag**2, dim=0, keepdim=True) + 1e-12
        return x_mag / torch.sqrt(power)

# =========================================================
# load encoder από checkpoint
# =========================================================
def load_encoder_from_checkpoint(pth_path, M=4):
    ckpt = torch.load(pth_path, map_location=device)

    enc = HybridEncoderURS(M=M, n=2, L=3).to(device)

    enc_state = {
        k.replace("enc2.", ""): v
        for k, v in ckpt.items()
        if k.startswith("enc2.")
    }

    enc.load_state_dict(enc_state)
    enc.eval()
    return enc

# =========================================================
# extract constellation / i-matrix
# =========================================================
def extract_imatrix(pth_path, tag):
    M = 4
    enc = load_encoder_from_checkpoint(pth_path, M)

    symbols = torch.arange(M, device=device)

    with torch.no_grad():
        x = enc(symbols)      # (M,1)

    x_np = x.cpu().numpy().squeeze()

    np.savez(
        f"urqswipt_imatrix_{tag}.npz",
        M=M,
        symbols=np.arange(M),
        x=x_np
    )

    print(f"[OK] Saved urqswipt_imatrix_{tag}.npz")
    print("Constellation magnitude:", x_np)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    extract_imatrix("autoenc2_4PAM_Pt10dBm.pth", tag="Pt10dBm")
    extract_imatrix("autoenc2_4PAM_Pt5dBm.pth", tag="Pt5dBm")
    extract_imatrix("autoenc2_4PAM_Pt0dBm.pth", tag="Pt0dBm")
    extract_imatrix("autoenc2_4PAM_Pt-5dBm.pth", tag="Pt-5dBm")
    extract_imatrix("autoenc2_4PAM_Pt-10dBm.pth", tag="Pt-10dBm")
    extract_imatrix("autoenc2_4PAM_Pt-15dBm.pth", tag="Pt-15dBm")
    extract_imatrix("autoenc2_4PAM_Pt-20dBm.pth", tag="Pt-20dBm")