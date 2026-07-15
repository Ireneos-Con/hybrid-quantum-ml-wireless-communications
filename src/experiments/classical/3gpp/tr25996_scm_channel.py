import math

import torch
import torch.nn as nn


def ebno_db2sigma(ebno_db, n, k):
    ebno_lin = 10.0 ** (ebno_db / 10.0)
    return math.sqrt(n / (2.0 * k * ebno_lin))


def signed_subpath_offsets(abs_offsets):
    """TR 25.996 Table 5.2 lists positive offsets for subpath pairs."""
    signed = []
    for value in abs_offsets:
        signed.extend([value, -value])
    return signed


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
        bs_as_deg = 15.0 if self.urban_macro_as_deg == 15.0 else 8.0
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
            raw_aod = (
                torch.randn(batch, self.N, device=dev, dtype=dtype)
                * params["r_as"]
                * params["bs_as_deg"]
            )
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
        _, aoa_paths = self._sample_path_angles(batch, powers, dev, dtype, params)
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
            shadow_db = torch.randn(batch, 1, device=dev, dtype=dtype) * params["shadow_std_db"]
            bulk_gain = torch.pow(
                torch.tensor(10.0, device=dev, dtype=dtype),
                -(params["pathloss_db"] + shadow_db) / 20.0,
            )
            hr = hr * bulk_gain
            hi = hi * bulk_gain
        if self.normalize_fading:
            scale = torch.rsqrt(torch.mean(hr**2 + hi**2).clamp_min(1e-12))
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
