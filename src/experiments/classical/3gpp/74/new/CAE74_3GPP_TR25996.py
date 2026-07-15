import sys
assert sys.version_info >= (3, 8)

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

batch_size = 32
steps_per_epoch = 1000
epochs = 80
EBNO_TRAIN_DB = 10.0
runs = 4


# =========================================================
# Utilities
# =========================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def ebno_db2sigma(ebno_db, n, k):
    ebno_lin = 10.0 ** (ebno_db / 10.0)
    return math.sqrt(n / (2.0 * k * ebno_lin))


def signed_subpath_offsets(abs_offsets):
    """TR 25.996 Table 5.2 lists positive offsets for subpath pairs."""
    signed = []
    for value in abs_offsets:
        signed.extend([value, -value])
    return signed


# =========================================================
# Encoder: M -> 2n (same CAE architecture as CAE74_3GPP.py)
# =========================================================
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
        z = z * math.sqrt(self.n) / torch.sqrt(power)
        return z


# =========================================================
# 3GPP TR 25.996 SCM effective block-fading channel
# =========================================================
class TR25996SCMEQ(nn.Module):
    """Effective SISO/block-fading channel derived from 3GPP TR 25.996 SCM.

    TR 25.996 defines delayed MIMO path matrices. The HAE/CAE paper uses the
    block-fading form y = h x + w with perfect equalization x_hat = y/h.
    This layer generates an SCM-based scalar h by summing the 6 paths and 20
    subpaths/path, then applies the same equalized channel used by the paper.
    """

    TABLE_52_BS_MACRO_OFFSETS_DEG = signed_subpath_offsets([
        0.0894, 0.2826, 0.4984, 0.7431, 1.0257,
        1.3594, 1.7688, 2.2961, 3.0389, 4.3101,
    ])
    TABLE_52_BS_MICRO_OFFSETS_DEG = signed_subpath_offsets([
        0.2236, 0.7064, 1.2461, 1.8578, 2.5642,
        3.3986, 4.4220, 5.7403, 7.5974, 10.7753,
    ])
    TABLE_52_MS_OFFSETS_DEG = signed_subpath_offsets([
        1.5649, 4.9447, 8.7224, 13.0045, 17.9492,
        23.7899, 30.9538, 40.1824, 53.1816, 75.4274,
    ])

    def __init__(
        self,
        k,
        n,
        ebno_db,
        scenario="urban_macro",
        velocity_kmh=30.0,
        fc_hz=1.9e9,
        distance_m=500.0,
        sample_time_s=1e-3,
        baseband_frequency_hz=0.0,
        normalize_fading=True,
        apply_pathloss=False,
        urban_macro_as_deg=8.0,
    ):
        super().__init__()
        self.k = k
        self.n = n
        self.ebno_db = ebno_db

        self.scenario = scenario
        self.N = 6
        self.num_subpaths = 20
        self.velocity_mps = velocity_kmh / 3.6
        self.fc_hz = fc_hz
        self.distance_m = distance_m
        self.sample_time_s = sample_time_s
        self.baseband_frequency_hz = baseband_frequency_hz
        self.normalize_fading = normalize_fading
        self.apply_pathloss = apply_pathloss
        self.urban_macro_as_deg = urban_macro_as_deg

        self.c = 3e8
        self.wavelength_m = self.c / self.fc_hz
        self.max_doppler_hz = self.velocity_mps / self.wavelength_m

    def _scenario_params(self):
        if self.scenario == "suburban_macro":
            return {
                "ds_mu_log10": -6.80,
                "ds_sigma_log10": 0.288,
                "r_ds": 1.4,
                "r_as": 1.2,
                "bs_as_deg": 5.0,
                "shadow_std_db": 8.0,
                "pathloss_db": 31.5 + 35.0 * math.log10(max(self.distance_m, 35.0)),
                "bs_offsets": self.TABLE_52_BS_MACRO_OFFSETS_DEG,
                "micro": False,
            }
        if self.scenario == "urban_micro":
            distance_m = max(self.distance_m, 20.0)
            return {
                "bs_as_deg": 19.0,
                "shadow_std_db": 10.0,
                "pathloss_db": 34.53 + 38.0 * math.log10(distance_m),
                "bs_offsets": self.TABLE_52_BS_MICRO_OFFSETS_DEG,
                "micro": True,
            }

        if self.urban_macro_as_deg == 15.0:
            bs_as_deg = 15.0
        else:
            bs_as_deg = 8.0

        return {
            "ds_mu_log10": -6.18,
            "ds_sigma_log10": 0.18,
            "r_ds": 1.7,
            "r_as": 1.3,
            "bs_as_deg": bs_as_deg,
            "shadow_std_db": 8.0,
            "pathloss_db": 34.5 + 35.0 * math.log10(max(self.distance_m, 35.0)),
            "bs_offsets": self.TABLE_52_BS_MACRO_OFFSETS_DEG,
            "micro": False,
        }

    def _sample_path_delays_and_powers(self, batch, dev, dtype, params):
        if params["micro"]:
            delays = torch.rand(batch, self.N, device=dev, dtype=dtype) * 1.2e-6
            delays = delays - delays.min(dim=1, keepdim=True).values
            shadow = torch.randn(batch, self.N, device=dev, dtype=dtype) * 3.0
            powers = torch.exp(-delays / 1e-6) * torch.pow(10.0, -shadow / 10.0)
        else:
            z_ds = torch.randn(batch, 1, device=dev, dtype=dtype)
            delay_spread = torch.pow(
                torch.tensor(10.0, device=dev, dtype=dtype),
                params["ds_mu_log10"] + params["ds_sigma_log10"] * z_ds,
            )
            u = torch.rand(batch, self.N, device=dev, dtype=dtype).clamp_min(1e-12)
            raw_delays = -params["r_ds"] * delay_spread * torch.log(u)
            raw_delays, _ = torch.sort(raw_delays, dim=1)
            delays = raw_delays - raw_delays[:, :1]

            shadow = torch.randn(batch, self.N, device=dev, dtype=dtype) * 3.0
            envelope = torch.exp(
                -raw_delays * (params["r_ds"] - 1.0)
                / (params["r_ds"] * delay_spread + 1e-18)
            )
            powers = envelope * torch.pow(10.0, -shadow / 10.0)

        powers = powers / (powers.sum(dim=1, keepdim=True) + 1e-12)
        return delays, powers

    def _sample_path_angles(self, batch, powers, dev, dtype, params):
        if params["micro"]:
            aod_paths = (torch.rand(batch, self.N, device=dev, dtype=dtype) * 80.0) - 40.0
        else:
            bs_std = params["r_as"] * params["bs_as_deg"]
            raw_aod = torch.randn(batch, self.N, device=dev, dtype=dtype) * bs_std
            angle_order = torch.argsort(torch.abs(raw_aod), dim=1)
            sorted_by_abs = torch.gather(raw_aod, 1, angle_order)
            power_order = torch.argsort(powers, dim=1, descending=True)
            aod_paths = torch.zeros_like(raw_aod)
            aod_paths.scatter_(1, power_order, sorted_by_abs)

        raw_aoa = torch.randn(batch, self.N, device=dev, dtype=dtype) * 68.0
        angle_order = torch.argsort(torch.abs(raw_aoa), dim=1)
        sorted_by_abs = torch.gather(raw_aoa, 1, angle_order)
        power_order = torch.argsort(powers, dim=1, descending=True)
        aoa_paths = torch.zeros_like(raw_aoa)
        aoa_paths.scatter_(1, power_order, sorted_by_abs)
        return aod_paths, aoa_paths

    def sample_channel_coefficients(self, batch, dev, dtype):
        params = self._scenario_params()
        delays, powers = self._sample_path_delays_and_powers(batch, dev, dtype, params)
        aod_paths, aoa_paths = self._sample_path_angles(batch, powers, dev, dtype, params)

        bs_offsets = torch.tensor(
            params["bs_offsets"], device=dev, dtype=dtype
        ).view(1, 1, self.num_subpaths)
        ms_offsets = torch.tensor(
            self.TABLE_52_MS_OFFSETS_DEG, device=dev, dtype=dtype
        ).view(1, 1, self.num_subpaths)

        pairing = torch.argsort(
            torch.rand(batch, self.N, self.num_subpaths, device=dev, dtype=dtype),
            dim=2,
        )
        ms_offsets = ms_offsets.expand(batch, self.N, self.num_subpaths)
        ms_offsets = torch.gather(ms_offsets, 2, pairing)

        aoa = aoa_paths.unsqueeze(2) + ms_offsets

        phases = 2.0 * math.pi * torch.rand(
            batch, self.N, self.num_subpaths, device=dev, dtype=dtype
        )
        velocity_angle = 2.0 * math.pi * torch.rand(batch, 1, 1, device=dev, dtype=dtype)
        time_s = self.sample_time_s * torch.rand(batch, 1, 1, device=dev, dtype=dtype)

        aoa_rad = torch.deg2rad(aoa)
        doppler_phase = (
            2.0
            * math.pi
            * self.max_doppler_hz
            * torch.cos(aoa_rad - velocity_angle)
            * time_s
        )
        delay_phase = -2.0 * math.pi * self.baseband_frequency_hz * delays.unsqueeze(2)

        path_amp = torch.sqrt(powers.unsqueeze(2) / self.num_subpaths)
        total_phase = phases + doppler_phase + delay_phase

        hr = torch.sum(path_amp * torch.cos(total_phase), dim=(1, 2)).view(batch, 1)
        hi = torch.sum(path_amp * torch.sin(total_phase), dim=(1, 2)).view(batch, 1)

        if self.apply_pathloss:
            shadow_db = (
                torch.randn(batch, 1, device=dev, dtype=dtype) * params["shadow_std_db"]
            )
            bulk_gain = torch.pow(
                torch.tensor(10.0, device=dev, dtype=dtype),
                -(params["pathloss_db"] + shadow_db) / 20.0,
            )
            hr = hr * bulk_gain
            hi = hi * bulk_gain

        if self.normalize_fading:
            mean_power = torch.mean(hr**2 + hi**2).clamp_min(1e-12)
            scale = torch.rsqrt(mean_power)
            hr = hr * scale
            hi = hi * scale

        return hr, hi

    def forward(self, z):
        batch = z.size(0)
        sigma = ebno_db2sigma(self.ebno_db, self.n, self.k)

        xr = z[:, :self.n]
        xi = z[:, self.n:]

        hr, hi = self.sample_channel_coefficients(batch, z.device, z.dtype)

        yr = hr * xr - hi * xi + sigma * torch.randn_like(xr)
        yi = hr * xi + hi * xr + sigma * torch.randn_like(xi)

        h2 = hr**2 + hi**2 + 1e-12
        xr_eq = (yr * hr + yi * hi) / h2
        xi_eq = (yi * hr - yr * hi) / h2

        return torch.cat([xr_eq, xi_eq], dim=1)


# =========================================================
# Decoder: 2n -> M
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
        self.chan = TR25996SCMEQ(
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


# =========================================================
# BLER vs SNR - CAE
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
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=torch.int64, device=device)

    msgs = torch.arange(16, device=device)
    bits = ((msgs[:, None] >> torch.tensor([3, 2, 1, 0], device=device)) & 1)
    codewords = (bits @ G) & 1
    return 1.0 - 2.0 * codewords.float()


@torch.no_grad()
def test_hamming74_mld_3gpp_tr25996(snr_dBs, Ntest):
    codebook = hamming74_codebook().to(device)
    bler = []

    for ebdb in snr_dBs:
        channel = TR25996SCMEQ(
            k,
            n,
            ebdb,
            scenario="urban_macro",
            velocity_kmh=30.0,
            fc_hz=1.9e9,
            distance_m=500.0,
            sample_time_s=1e-3,
            normalize_fading=True,
            apply_pathloss=False,
            urban_macro_as_deg=8.0,
        ).to(device)

        syms = torch.randint(0, 16, (Ntest,), device=device)
        tx = codebook[syms]

        xr = tx
        xi = torch.zeros_like(tx)
        z = torch.cat([xr, xi], dim=1)

        y_eq = channel(z)
        yr = y_eq[:, :n]

        d2 = ((yr[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2)
        preds = d2.argmin(dim=1)

        bler.append((preds != syms).float().mean().item())

    return np.array(bler)


# =========================================================
# Run evaluation + SAVE
# =========================================================
Ntest = 100_000
snr_dBs = np.linspace(0, 20, 21)
syms_np = np.random.randint(0, M, size=Ntest)

bler_cae = avg_runs(
    test_fn=test_bler_cae,
    runs=runs,
    autoenc=model,
    syms_np=syms_np,
    snr_dBs=snr_dBs,
)

bler_hamming = avg_runs(
    test_fn=test_hamming74_mld_3gpp_tr25996,
    runs=runs,
    snr_dBs=snr_dBs,
    Ntest=Ntest,
)

np.savez(
    "CAE_74_vs_Hamming_3GPP_TR25996.npz",
    snr_dBs=snr_dBs.astype(np.float32),
    bler_cae_3gpp_tr25996=bler_cae.astype(np.float32),
    bler_hamming_3gpp_tr25996=bler_hamming.astype(np.float32),
    k=np.int32(k),
    n=np.int32(n),
    ebno_train_db=np.float32(EBNO_TRAIN_DB),
    scenario="urban_macro",
    velocity_kmh=np.float32(30.0),
    fc_hz=np.float32(1.9e9),
    distance_m=np.float32(500.0),
    normalized_fading=np.bool_(True),
)

print("Saved: CAE_74_vs_Hamming_3GPP_TR25996.npz")
