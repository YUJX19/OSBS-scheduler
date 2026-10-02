"""OSBS inference network, causal input transforms, and the repetition rule.

``decide`` evaluates episode--window pairs from observations.h5 using delayed CSI, navigation state, re-aim timing, P0, protocol MCS, and requested-window time. It does not open targets, labels, or future horizons. ``load_future`` loads evaluator-only future data, and ``served_bler_curve`` evaluates a candidate decision against that data.

For one episode, ``Z_dd`` is [M, N, 3], ``csi_field_db`` is [3, M, N] with the latest field first, ``csi_age_ms`` is [3], ``beam_phase_ms`` is [2], ``p0_db`` is scalar, ``z_nav`` is [13], and ``mu_p`` is the protocol MCS. ``decide`` returns a Boolean candidate served-bin mask and integer K in 1..8, or K=0 for a skipped window. A nonempty candidate mask with K=0 does not represent a transmission. Callers may supply an actual budget and a validation-selected repetition increment; both default to the checkpoint budget and zero for backward compatibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Variant -> (use_film, use_residual).
VARIANTS = {"nonav": (False, False), "film": (True, False), "res": (False, True), "hybrid": (True, True)}
SHARED = ("encoder", "skip", "level_head")       # Modules copied from the non-navigation reference at paired initialisation.
P0_REF_DB, P0_SCALE_DB = -50.0, 6.0               # power-offset input channel: (P0 - P0_REF_DB) / P0_SCALE_DB
# Timing shared by the released corpora, in ms (manifest physics.beams, physics.csi, and physics.horizon).
DECISION_MS, RESTEER_MS, LATENCY_MS = 980, 1000, 20
CSI_PERIOD_MS, CSI_FIELDS, CSI_AGES_MS = 120, 3, (120, 240, 360, 480)
WINDOWS, WINDOW_SPACING_MS, REPETITIONS, REPETITION_SPACING_MS = 12, 80, 8, 1
CSI_AGE_REF_MS, CSI_AGE_SCALE_MS = 300.0, 180.0   # Latest-age input channel: (age - 300) / 180 spans -1..1.
# Stored observation stack: clipped dB / 20, ln(1 + gamma), and negated z-score of ln(1 + gamma).
FEATURE_DB_LO, FEATURE_DB_HI, FEATURE_DB_SCALE, WEAKNESS_STD_FLOOR = -30.0, 60.0, 20.0, 1.0e-4
INPUT_CHANNELS = 16               # Fifteen per-episode channels and one requested-window channel.
OBSERVATION_FIELDS = ("Z_dd", "csi_field_db", "csi_age_ms", "beam_phase_ms", "p0_db", "z_nav", "mu_p")
CSI_SOURCES = ("feedback", "measured")    # Fed-back block CSI from observations.h5 or per-bin measurements from csi_measurement.h5.
NAV_GEOMETRY = ("distance_m", "log10_distance", "radial_speed", "transverse_speed", "relative_speed", "height_difference_m",
                "tx_misalignment_csi", "rx_misalignment_csi", "tx_misalignment_window", "rx_misalignment_window",
                "log10_distance_ratio")
LEVEL_QUANTILES = (0.5, 0.2)     # Level-head outputs: median and 0.2 quantile of the level change.
LEVEL_SCALE_DB = 3.5             # Level-head output scale in dB.


def quality_features(csi_db):
    """Transform dB CSI arrays ``[..., M, N]`` into ``[..., M, N, 3]`` observation features.

    The features are clipped dB / 20, ln(1 + linear SINR), and its negated fieldwise z-score. The z-score is zero when the sample standard deviation is below 1e-4.
    """
    db = np.asarray(csi_db, dtype=np.float64)
    if db.ndim < 2 or not np.isfinite(db).all():
        raise ValueError("CSI fields must be finite [..., M, N] dB arrays")
    gamma = 10.0 ** (db / 10.0)
    capacity = np.log1p(gamma)
    std = capacity.std(axis=(-2, -1), ddof=1, keepdims=True)
    weakness = np.where(std >= WEAKNESS_STD_FLOOR,
                        -(capacity - capacity.mean(axis=(-2, -1), keepdims=True)) / np.maximum(std, WEAKNESS_STD_FLOOR),
                        0.0)
    return np.stack((np.clip(db, FEATURE_DB_LO, FEATURE_DB_HI) / FEATURE_DB_SCALE, capacity, weakness),
                    axis=-1).astype(np.float32)


def reaim_timing_ms(beam_phase_ms):
    """Time since the last and until the next re-aim of each end, in ms at the decision subframe: [E, 2 ends, 2].

    End s is aimed at subframe 0 without delay and re-aimed by commands at subframes phase[s] + RESTEER_MS k (k >= 0)
    that take effect LATENCY_MS later.  At the decision subframe the beam in force dates from phase + LATENCY_MS when
    phase > 0 and that instant is not after the decision subframe, otherwise from subframe 0; the next re-aim takes
    effect at phase + LATENCY_MS when that is after the decision subframe, otherwise one re-aim period later."""
    phase = np.asarray(beam_phase_ms, dtype=np.int64)
    if phase.ndim != 2 or phase.shape[1] != 2 or np.any((phase < 0) | (phase >= RESTEER_MS)):
        raise ValueError("beam_phase_ms must be [E, 2] offsets in 0..RESTEER_MS-1")
    effect = phase + LATENCY_MS
    last_effect = np.where((phase > 0) & (effect <= DECISION_MS), effect, 0)
    next_effect = np.where(effect > DECISION_MS, effect, effect + RESTEER_MS)
    return np.stack((DECISION_MS - last_effect, next_effect - DECISION_MS), axis=-1).astype(np.float32)


def window_time_ms(windows):
    """Centre of the repetition subframes of window w in ms after the decision subframe: WINDOW_SPACING_MS w + 4.5
    (the 8 repetitions occupy the consecutive subframes 1..8 ms after the window start)."""
    w = np.asarray(windows, dtype=np.int64)
    if np.any((w < 0) | (w >= WINDOWS)):
        raise ValueError("window index must be in 0..WINDOWS-1")
    first, last = REPETITION_SPACING_MS, REPETITIONS * REPETITION_SPACING_MS
    return (WINDOW_SPACING_MS * w + 0.5 * (first + last)).astype(np.float32)


def _planes(values, spatial):
    """Constant channels [E, M, N, k] from per-episode values [E, k]."""
    v = np.asarray(values, dtype=np.float32)
    return np.broadcast_to(v[:, None, None, :], spatial + (v.shape[-1],))


def network_input(obs, p0_offset_db=0.0):
    """Build 15 causal per-episode channels with shape ``[E, M, N, 15]`` from ``obs``.

    ``obs`` contains ``Z_dd [E, M, N, 3]``, three CSI fields ``[E, 3, M, N]``, CSI ages ``[E, 3]``, re-aim phases ``[E, 2]``, and one P0 value per episode. ``with_window_plane`` adds the requested-window channel. ``p0_offset_db`` is an optional caller-supplied offset applied before P0 scaling.
    """
    z = np.asarray(obs["Z_dd"], dtype=np.float32)
    p0 = np.asarray(obs["p0_db"], dtype=np.float32).reshape(-1) + float(p0_offset_db)
    if z.ndim != 4 or z.shape[-1] != 3 or p0.shape != (z.shape[0],) or not np.isfinite(p0).all():
        raise ValueError("network_input needs Z_dd [E, M, N, 3] and one finite P0 per episode")
    spatial = z.shape[:3]
    p0_plane = _planes(((p0 - P0_REF_DB) / P0_SCALE_DB)[:, None], spatial)
    csi = np.asarray(obs["csi_field_db"], dtype=np.float32)
    age = np.asarray(obs["csi_age_ms"], dtype=np.float32)
    if csi.shape != (len(z), CSI_FIELDS) + spatial[1:] or age.shape != (len(z), CSI_FIELDS):
        raise ValueError("network_input needs csi_field_db [E, 3, M, N] and csi_age_ms [E, 3]")
    older = quality_features(csi[:, 1:])                                            # [E, 2, M, N, 3]
    older = np.moveaxis(older, 1, 3).reshape(spatial + (2 * 3,))                   # channels of field 1 then field 2
    timing = reaim_timing_ms(obs["beam_phase_ms"]).reshape(len(z), 4) / float(RESTEER_MS)
    scalars = np.concatenate((((age[:, :1] - CSI_AGE_REF_MS) / CSI_AGE_SCALE_MS), timing), axis=1)
    return np.concatenate((z, older, p0_plane, _planes(scalars, spatial)), axis=-1)


def with_window_plane(Z, windows):
    """Append the scheduled window's time channel, (t_w - 480) / 480, to network_input channels.

    Z is [B, M, N, 15] (numpy or torch); windows are the B window indices.  Returns [B, M, N, 16]."""
    t = (window_time_ms(np.asarray(windows).reshape(-1)) - 480.0) / 480.0
    if isinstance(Z, torch.Tensor):
        plane = torch.as_tensor(t, dtype=Z.dtype, device=Z.device)[:, None, None, None].expand(-1, Z.shape[1], Z.shape[2], 1)
        return torch.cat((Z, plane), dim=-1)
    Z = np.asarray(Z, dtype=np.float32)
    return np.concatenate((Z, _planes(t[:, None], Z.shape[:3])), axis=-1)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _angle(a, b):
    return np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1.0, 1.0))


def navigation_geometry(obs) -> np.ndarray:
    """Build 11 deterministic geometry features for every episode and requested window, shape ``[E, W, 11]``.

    ``z_nav`` supplies relative position in m, both velocities in m/s, and both commanded boresights in rad. The function extrapolates position at relative velocity to the latest CSI instant and requested-window centre. It returns decision-time distance, log distance, radial, transverse and total relative speed, and height difference, followed by two beam/line-of-sight angles at the CSI instant, two at the requested window, and the log distance ratio. Beam directions follow the re-aim schedule. This calculation uses no future CSI, targets, labels, antenna constants, or propagation constants.
    """
    nav = np.asarray(obs["z_nav"], dtype=np.float64)
    age = np.asarray(obs["csi_age_ms"], dtype=np.float64)[:, 0]
    timing = reaim_timing_ms(obs["beam_phase_ms"]).astype(np.float64)               # [E, 2 ends, since/until]
    rel, vrel = nav[:, 0:3], nav[:, 6:9] - nav[:, 3:6]
    distance = np.linalg.norm(rel, axis=1)
    u = rel / distance[:, None]
    radial = (vrel * u).sum(1)
    transverse = np.linalg.norm(vrel - radial[:, None] * u, axis=1)

    def aim(az, el):
        return np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1)

    commanded = np.stack((aim(nav[:, 9], nav[:, 10]), aim(nav[:, 11], nav[:, 12])), axis=1)     # [E, 2, 3]

    def rel_at(t):
        return rel + vrel * (np.asarray(t, dtype=np.float64)[..., None] / 1000.0)

    def misalignment(t):
        """[E, 2] angle of each end's beam in force to the line of sight at t [E] (ms from the decision subframe)."""
        angles = np.empty((len(nav), 2))
        for s, sign in ((0, 1.0), (1, -1.0)):
            since, until = timing[:, s, 0], timing[:, s, 1]
            later, earlier = t >= until, t < -since
            t_reaim = until + np.floor(np.maximum(t - until, 0.0) / RESTEER_MS) * RESTEER_MS
            previous = -since - RESTEER_MS
            previous_cmd = np.where(previous < -DECISION_MS, -DECISION_MS, previous - LATENCY_MS)
            direction = np.where(later[:, None], sign * rel_at(t_reaim - LATENCY_MS),
                                 np.where(earlier[:, None], sign * rel_at(previous_cmd), commanded[:, s]))
            angles[:, s] = _angle(direction, sign * rel_at(t))
        return angles

    t_csi = -age
    d_csi = np.linalg.norm(rel_at(t_csi), axis=1)
    at_csi = misalignment(t_csi)
    out = np.empty((len(nav), WINDOWS, len(NAV_GEOMETRY)), dtype=np.float32)
    for w in range(WINDOWS):
        t = np.full(len(nav), float(window_time_ms(w)))
        at_window = misalignment(t)
        out[:, w] = np.stack((distance, np.log10(distance), radial, transverse, np.linalg.norm(vrel, axis=1), rel[:, 2],
                              at_csi[:, 0], at_csi[:, 1], at_window[:, 0], at_window[:, 1],
                              np.log10(np.linalg.norm(rel_at(t), axis=1) / d_csi)), axis=1)
    return out


def navigation_input(z_nav, geometry, nav_mean, nav_std, window):
    """Standardised navigation input of one window for every episode: z_nav [E, 13] followed by that window's link
    geometry (geometry [E, W, 11]); nav_mean / nav_std are the checkpoint's statistics."""
    x = np.concatenate((np.asarray(z_nav, dtype=np.float32), np.asarray(geometry, dtype=np.float32)[:, window]), axis=1)
    mean, std = (np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v, dtype=np.float32) for v in (nav_mean, nav_std))
    if mean.shape != (x.shape[1],) or std.shape != (x.shape[1],):
        raise ValueError(f"navigation statistics have {mean.shape[0]} entries; the input has {x.shape[1]}")
    return (x - mean) / std


def load_observations(root, ids=None, p0_index=0, csi="feedback"):
    """Load selected causal observations at one P0-axis entry.

    ``csi="feedback"`` reads fed-back block CSI from observations.h5. ``csi="measured"`` replaces those CSI fields and ``Z_dd`` with csi_measurement.h5 values while retaining the fed-back CSI's protocol MCS. This function does not open targets, labels, or future horizons.
    """
    if csi not in CSI_SOURCES:
        raise ValueError(f"unknown CSI source {csi!r}")
    import h5py
    if ids is None:
        sel, order = slice(None), None
    else:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        order = np.argsort(ids, kind="stable")                    # h5py reads increasing indices; reordered below
        sel = ids[order]
    with h5py.File(Path(root) / "observations.h5", "r") as f:
        obs = {"Z_dd": np.asarray(f["Z_dd"][sel, p0_index], dtype=np.float32),
               "csi_field_db": np.asarray(f["csi_field_db"][sel, p0_index], dtype=np.float32),
               "csi_age_ms": np.asarray(f["csi_age_ms"][sel], dtype=np.int64),
               "beam_phase_ms": np.asarray(f["beam_phase_ms"][sel], dtype=np.int64),
               "p0_db": np.asarray(f["p0_db"][sel, p0_index], dtype=np.float32),
               "z_nav": np.asarray(f["z_nav"][sel], dtype=np.float32),
               "mu_p": np.asarray(f["mu_p"][sel, p0_index], dtype=np.int64)}
        if csi == "measured":
            with h5py.File(Path(root) / "csi_measurement.h5", "r") as m:
                obs["csi_field_db"] = np.asarray(m["csi_measured_db"][sel, p0_index], dtype=np.float32)
            obs["Z_dd"] = quality_features(obs["csi_field_db"][:, 0])
    if order is not None:
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        obs = {k: v[inverse] for k, v in obs.items()}
    return obs



class _ResidualBlock(nn.Module):
    """ResNet basic block: 3x3 conv-BN-ReLU, 3x3 conv-BN, shortcut sum, ReLU; with stride 2 the first conv and the
    shortcut (1x1 conv-BN) downsample."""

    def __init__(self, d: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(d, d, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(d)
        if stride != 1:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv2d(d, d, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(d),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class DdEncoder(nn.Module):
    """3x3 conv-BN-ReLU stem and n_blocks residual blocks, the first with stride 2:
    [B, M, N, C_dd] -> [B, M', N', d] with M' = M/2, N' = N/2 for even M, N."""

    def __init__(self, C_dd: int, d: int, n_blocks: int = 3):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.C_dd = C_dd
        self.d = d
        self.n_blocks = n_blocks
        self.stem = nn.Sequential(
            nn.Conv2d(C_dd, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            stride = 2 if i == 0 else 1
            blocks.append(_ResidualBlock(d, stride=stride))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, Z_dd: Tensor) -> Tensor:
        """Z_dd: [B, M, N, C_dd] -> [B, M', N', d]."""
        if Z_dd.dim() != 4:
            raise ValueError(f"Z_dd must be [B, M, N, C_dd]; got {tuple(Z_dd.shape)}")
        x = Z_dd.permute(0, 3, 1, 2).contiguous()   # [B, C_dd, M, N]
        x = self.stem(x)
        x = self.blocks(x)
        return x.permute(0, 2, 3, 1).contiguous()   # [B, M', N', d]


class FiLMGenerator(nn.Module):
    """MLP navigation input -> per-channel FiLM scale alpha and shift delta.

    The output layers are initialised to alpha = 1, delta = 0, so the modulation starts as the identity.
    """

    def __init__(self, nav_dim: int, d: int, hidden: int = 64):
        super().__init__()
        self.nav_dim = nav_dim
        self.d = d
        self.body = nn.Sequential(
            nn.Linear(nav_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.to_alpha = nn.Linear(hidden, d)
        self.to_delta = nn.Linear(hidden, d)
        nn.init.zeros_(self.to_alpha.weight)
        nn.init.ones_(self.to_alpha.bias)         # alpha = 1 at init
        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)        # delta = 0 at init

    def forward(self, z_nav: Tensor) -> tuple[Tensor, Tensor]:
        """z_nav: [B, nav_dim] -> (alpha [B, d], delta [B, d])."""
        h = self.body(z_nav)
        return self.to_alpha(h), self.to_delta(h)


def film_apply(F: Tensor, alpha: Tensor, delta: Tensor) -> Tensor:
    """F_tilde = alpha (.) F + delta, broadcast over spatial dims.

    Args:
        F:     [B, M', N', d]
        alpha: [B, d]
        delta: [B, d]
    """
    if F.dim() != 4 or alpha.dim() != 2 or delta.dim() != 2:
        raise ValueError("F must be [B,M',N',d]; alpha/delta must be [B, d]")
    return alpha[:, None, None, :] * F + delta[:, None, None, :]


class SkipDecoder(nn.Module):
    """Bilinearly upsample the modulated feature map to the M x N grid and project every bin to one skip logit (1x1
    conv).  Returns the logits [B, M, N] and the upsampled features [B, M, N, d] that the served-bin pooling reads."""

    def __init__(self, d: int, M: int, N: int):
        super().__init__()
        self.M = M
        self.N = N
        self.proj = nn.Conv2d(d, 1, kernel_size=1)

    def forward(self, F_tilde: Tensor) -> tuple[Tensor, Tensor]:
        if F_tilde.dim() != 4:
            raise ValueError(f"F_tilde must be [B, M', N', d]; got {tuple(F_tilde.shape)}")
        x = F_tilde.permute(0, 3, 1, 2).contiguous()                   # [B, d, M', N']
        x_up = F.interpolate(x, size=(self.M, self.N), mode="bilinear", align_corners=False)
        return self.proj(x_up)[:, 0], x_up.permute(0, 2, 3, 1).contiguous()


class LevelHead(nn.Module):
    """Pooled descriptor [..., d] -> level change of the scheduled window in dB at LEVEL_QUANTILES [..., 2].

    The median is LEVEL_SCALE_DB times the first output; the 0.2-quantile lies below it by LEVEL_SCALE_DB times the
    softplus of the second, so the two quantiles never cross.  raw() gives the untransformed outputs, to which the
    residual level head adds its correction."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2))

    def raw(self, u: Tensor) -> Tensor:
        return self.mlp(u)

    @staticmethod
    def transform(raw: Tensor) -> Tensor:
        median = LEVEL_SCALE_DB * raw[..., 0]
        return torch.stack((median, median - LEVEL_SCALE_DB * F.softplus(raw[..., 1])), dim=-1)


class ResidualLevelHead(nn.Module):
    """MLP on concat(stop_grad(u), navigation input) -> correction added to the level head's raw outputs.  The last
    layer is zero-initialised, so the correction starts at zero."""

    def __init__(self, d: int, nav_dim: int, hidden: int = 128):
        super().__init__()
        self.nav_dim = nav_dim
        self.mlp = nn.Sequential(nn.Linear(d + nav_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u_sg: Tensor, z_nav: Tensor) -> Tensor:
        """u_sg [B, d] (stop-grad), z_nav [B, nav_dim] -> [B, 2]."""
        return self.mlp(torch.cat([u_sg, z_nav], dim=-1))


@dataclass(frozen=True)
class OsbsConfig:
    use_film: bool
    use_residual: bool
    C_dd: int
    d: int
    M: int
    N: int
    nav_dim: int = 13            # navigation vector; the window's link geometry (NAV_GEOMETRY) follows it
    n_encoder_blocks: int = 3
    csi: str = "feedback"        # CSI the observations are built from (CSI_SOURCES)


class OsbsScheduler(nn.Module):
    """OSBS network of one variant (hybrid / film / res / nonav), selected by cfg.use_film / cfg.use_residual.

    forward(Z_dd, z_nav) returns the per-bin skip logits [B, M, N] and the served bins' level change [B, 2] (dB, at
    LEVEL_QUANTILES).  The pooling weights P(served) are detached, so the level terms reach the skip map only through
    the shared features.
    """

    def __init__(self, cfg: OsbsConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        nav_in = cfg.nav_dim + len(NAV_GEOMETRY)
        self.encoder = DdEncoder(cfg.C_dd, cfg.d, cfg.n_encoder_blocks)
        self.film: FiLMGenerator | None = FiLMGenerator(nav_in, cfg.d) if cfg.use_film else None
        self.skip = SkipDecoder(cfg.d, cfg.M, cfg.N)
        self.level_head = LevelHead(cfg.d)
        self.res_level: ResidualLevelHead | None = ResidualLevelHead(cfg.d, nav_in) if cfg.use_residual else None

    def forward(self, Z_dd: Tensor, z_nav: Tensor) -> tuple[Tensor, Tensor]:
        if Z_dd.dim() != 4:
            raise ValueError(f"Z_dd must be [B, M, N, C_dd]; got {tuple(Z_dd.shape)}")
        if z_nav.dim() != 2:
            raise ValueError(f"z_nav must be [B, nav_dim]; got {tuple(z_nav.shape)}")
        if Z_dd.shape[0] != z_nav.shape[0]:
            raise ValueError("batch size of Z_dd and z_nav must match")
        encoded = self.encoder(Z_dd)                               # [B, M', N', d]
        if self.film is not None:
            alpha, delta = self.film(z_nav)
            encoded = film_apply(encoded, alpha, delta)
        skip_logit, features = self.skip(encoded)                  # [B, M, N], [B, M, N, d]
        weight = torch.sigmoid(-skip_logit).detach().unsqueeze(-1)  # P(served)
        u = (weight * features).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-6)
        raw = self.level_head.raw(u)
        if self.res_level is not None:
            raw = raw + self.res_level(u.detach(), z_nav)
        return skip_logit, LevelHead.transform(raw)


def build_model(variant: str, cfg: dict, seed: int | None = None) -> OsbsScheduler:
    """Network of one variant.  With a seed, the initialisation is paired across variants: the nonav network is drawn
    first and its SHARED modules are copied into the variant, so all variants start from the same function (FiLM
    starts at the identity, the residual level head at zero)."""
    film, res = VARIANTS[variant]
    if seed is None:
        return OsbsScheduler(OsbsConfig(use_film=film, use_residual=res, **cfg))
    torch.manual_seed(seed)
    ref = OsbsScheduler(OsbsConfig(use_film=False, use_residual=False, **cfg))
    if variant == "nonav":
        return ref
    torch.manual_seed(seed)
    net = OsbsScheduler(OsbsConfig(use_film=film, use_residual=res, **cfg))
    for name in SHARED:
        getattr(net, name).load_state_dict(getattr(ref, name).state_dict())
    return net


def load_checkpoint(path, device="cpu"):
    """Trained checkpoint -> (network in eval mode on device, checkpoint dict).  The dict holds state_dict, nav_mean,
    nav_std (the standardisation of the navigation input used in training), variant, seed, cfg, epoch, rule_offset_db
    (the offset subtracted from the forecast 0.2-quantile in the repetition rule, chosen on validation),
    training_provenance and budget_provenance."""
    ck = torch.load(path, map_location="cpu")
    net = build_model(ck["variant"], ck["cfg"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), ck


@dataclass(frozen=True)
class BlerCurveParams:
    """Logistic BLER curve per MCS: slope a_mu_of_nb(mu, n_b) [1/dB] and midpoint b_mu_of_nb(mu, n_b) [dB]."""
    a_mu_of_nb: Callable[[int, int], float]
    b_mu_of_nb: Callable[[int, int], float]


# TS 38.214 Table 5.1.3.1-2, rows 0..22: (modulation order, code rate, spectral efficiency)
_NR_LADDER: tuple[tuple[int, float, float], ...] = (
    (2, 0.1172, 0.2344),
    (2, 0.1885, 0.3770),
    (2, 0.3008, 0.6016),
    (2, 0.4385, 0.8770),
    (2, 0.5879, 1.1758),
    (4, 0.3691, 1.4766),
    (4, 0.4238, 1.6953),
    (4, 0.4785, 1.9141),
    (4, 0.5400, 2.1602),
    (4, 0.6016, 2.4063),
    (4, 0.6426, 2.5703),
    (6, 0.4551, 2.7305),
    (6, 0.5049, 3.0293),
    (6, 0.5537, 3.3223),
    (6, 0.6016, 3.6094),
    (6, 0.6504, 3.9023),
    (6, 0.7021, 4.2129),
    (6, 0.7539, 4.5234),
    (6, 0.8027, 4.8164),
    (6, 0.8525, 5.1152),
    (8, 0.6665, 5.3320),
    (8, 0.6943, 5.5547),
    (8, 0.7363, 5.8906),
)

# Ten-percent BLER anchors and 10-to-1-percent gaps in dB for MCS 0..22; see the release DATA_CARD.
GAO2024_THR10_DB: tuple[float, ...] = (
    -6.02, -4.14, -2.05, -0.03, 1.99, 7.03, 9.93, 11.01, 11.95, 12.09, 13.10, 15.12,
    16.07, 19.03, 19.10, 21.06, 21.13, 23.02, 23.96, 24.09, 28.07, 31.10, 35.14,
)
GAP_10_TO_1_DB: tuple[float, ...] = (
    0.36, 0.38, 0.37, 0.33, 0.32, 0.31, 0.30, 0.30, 0.31, 0.32, 0.31, 0.28,
    0.30, 0.30, 0.27, 0.32, 0.32, 0.28, 0.29, 0.30, 0.27, 0.24, 0.28,
)
ANCHOR_BLER: float = 0.1             # the BLER at which GAO2024_THR10_DB is defined
GAP_BLER: float = 0.01               # the BLER that GAP_10_TO_1_DB reaches from ANCHOR_BLER
CODEWORD_RE: int = 512               # resource elements of the codeword the curve describes
CURVE_CONTRACT_SCHEMA: str = "bler_curve"
LINK_CONTRACT_SCHEMA: str = "link_contract"
LABEL_NUMERIC_CONTRACT: str = "float32_threshold_float16_horizon"
STORAGE_PRECISION: str = "csi_db_float32_threshold_db_float32_horizon_db_float16"
# Linear-scale EESM beta values for MCS 0..22 from 5G-LENA nr-eesm-t2.cc BetaTable2.
EESM_BETA_TABLE2: tuple[float, ...] = (1.60, 1.63, 1.67, 1.73, 1.79, 4.27, 4.71, 5.16, 5.66, 6.16, 6.50, 10.97,
                                       12.92, 14.96, 17.06, 19.33, 21.85, 24.51, 27.14, 29.94, 56.48, 65.00, 78.58)
# Nominal quantized CSI-value layout; it is not a complete signalling-packet specification.
CSI_FEEDBACK: dict = {"block": [1, 8], "value": "mean dB of the measured per-bin SINR of the block",
                      "strongest_block_range_db": [-23.0, 40.0], "strongest_block_step_db": 0.5,
                      "differential_step_db": 2.0, "differential_levels": 15, "bits": 259}
LINK_N_B: int = 512
MCS_SELECT_BLER: float = 0.1
REPETITION_TARGET_BLER: float = 0.01
LINK_K_MAX: int = 8
REPETITION_BUDGETS = (512, 1024, 1536, 2048, 4096)
INCREMENT_MAX = 7
EPS_TARGET: float = REPETITION_TARGET_BLER      # BLER target of the repetition rule and of the labels
# OTFS numerology (episode_physics.PhysicsConfig)
WAVEFORM: dict = {"delay_bins": 16, "doppler_bins": 32, "subcarrier_hz": 120_000.0, "cp_samples": 10, "carrier_hz": 28e9}


NUM_MCS: int = len(_NR_LADDER)
_DEFAULT_SE_TABLE: tuple[float, ...] = tuple(row[2] for row in _NR_LADDER)
assert len(GAO2024_THR10_DB) == NUM_MCS and len(GAP_10_TO_1_DB) == NUM_MCS and len(EESM_BETA_TABLE2) == NUM_MCS


def eesm_beta(mu) -> np.ndarray:
    """EESM beta (linear) of MCS index / indices mu."""
    index = np.asarray(mu, dtype=np.int64)
    if np.any((index < 0) | (index >= NUM_MCS)):
        raise ValueError("mu must be a 0..22 MCS index")
    return np.asarray(EESM_BETA_TABLE2, dtype=np.float64)[index]


def _require_codeword(n_b: int) -> None:
    if int(n_b) != CODEWORD_RE:
        raise ValueError(f"the BLER curve is defined for the {CODEWORD_RE}-RE codeword only, got n_b={n_b}")


def make_default_params() -> BlerCurveParams:
    """Logistic curve of every MCS through BLER(anchor) = ANCHOR_BLER and BLER(anchor + gap) = GAP_BLER."""
    logit_anchor = math.log((1.0 - ANCHOR_BLER) / ANCHOR_BLER)
    logit_gap = math.log((1.0 - GAP_BLER) / GAP_BLER)

    def a_fn(mu: int, n_b: int) -> float:
        _require_codeword(n_b)
        return (logit_gap - logit_anchor) / GAP_10_TO_1_DB[int(mu)]

    def b_fn(mu: int, n_b: int) -> float:
        return GAO2024_THR10_DB[int(mu)] - logit_anchor / a_fn(mu, n_b)

    return BlerCurveParams(a_mu_of_nb=a_fn, b_mu_of_nb=b_fn)


def gamma_threshold_db(mu: int, eps: float, n_b: int, params: BlerCurveParams) -> float:
    """SINR [dB] at which the BLER of MCS mu equals eps (closed-form inverse of the logistic)."""
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    a = float(params.a_mu_of_nb(int(mu), int(n_b)))
    b = float(params.b_mu_of_nb(int(mu), int(n_b)))
    return b + math.log((1.0 - eps) / eps) / a


def threshold_ladder_db(eps: float, n_b: int, params: BlerCurveParams) -> list[float]:
    """gamma_threshold_db for every MCS of the ladder."""
    return [gamma_threshold_db(mu, eps, n_b, params) for mu in range(NUM_MCS)]


def spectral_efficiency(mu: int) -> float:
    """Spectral efficiency [bit/RE] of MCS mu."""
    if not 0 <= mu < NUM_MCS:
        raise ValueError(f"mu={mu} out of range [0, {NUM_MCS})")
    return _DEFAULT_SE_TABLE[mu]



_CURVE = make_default_params()
_A = np.asarray([_CURVE.a_mu_of_nb(mu, LINK_N_B) for mu in range(NUM_MCS)], dtype=np.float64)
_B = np.asarray([_CURVE.b_mu_of_nb(mu, LINK_N_B) for mu in range(NUM_MCS)], dtype=np.float64)
_SE = np.asarray([spectral_efficiency(mu) for mu in range(NUM_MCS)], dtype=np.float64)
_BETA = np.asarray(EESM_BETA_TABLE2, dtype=np.float64)        # EESM beta of every MCS



def reliability_threshold_db(protocol_mu, epsilon: float = REPETITION_TARGET_BLER) -> np.ndarray:
    """Return the model BLER-curve threshold in dB for each protocol-MCS index and target epsilon.

    The repetition rule compares EESM effective SINR with this model threshold; it is not a per-decision reliability guarantee.
    """
    mu = np.asarray(protocol_mu, dtype=np.int64)
    if np.any((mu < 0) | (mu >= NUM_MCS)) or not 0.0 < epsilon < 1.0:
        raise ValueError("protocol_mu must be a 0..22 MCS index and epsilon in (0, 1)")
    return _B[mu] + math.log(1.0 / epsilon - 1.0) / _A[mu]


def served_rule_reach(gamma_lin, level_db, threshold_db, serve, beta, *, softness_db: float | None = None):
    """Evaluate K=1..K_max against the rule's model threshold for each served-bin group.

    The rule shifts latest observed linear SINR by ``level_db``, multiplies it by K, applies EESM with the protocol-MCS beta, and compares the result with ``threshold_db``. Inputs are ``gamma_lin [B, bins]``, ``level_db [B]`` or ``[B, R]``, ``threshold_db [B]``, and Boolean ``serve [B, bins]``. The result is ``[B, (R,) K_max]`` as hard 0/1 values, or logistic margins when ``softness_db`` is set. A row with no served bins never reaches.
    """
    scale = torch.pow(10.0, level_db / 10.0)
    extra = scale.dim() - 1
    K = torch.arange(1, LINK_K_MAX + 1, dtype=gamma_lin.dtype, device=gamma_lin.device)
    x = (scale[..., None] * K.view((1,) * scale.dim() + (LINK_K_MAX,)))[..., None] * gamma_lin.view(
        (gamma_lin.shape[0],) + (1,) * (extra + 1) + (gamma_lin.shape[1],))              # [B, (R,) K, bins]
    keep = serve.view((serve.shape[0],) + (1,) * (extra + 1) + (serve.shape[1],))
    count = serve.sum(-1).clamp_min(1).to(gamma_lin.dtype).view((-1,) + (1,) * (extra + 1))
    b = torch.as_tensor(beta, dtype=gamma_lin.dtype, device=gamma_lin.device).view((-1,) + (1,) * (extra + 1))
    lse = torch.logsumexp((-x / b[..., None]).masked_fill(~keep, -torch.inf), dim=-1)
    eff = -b * (lse - torch.log(count))
    margin = 10.0 * torch.log10(eff.clamp_min(1e-20)) - threshold_db.view((-1,) + (1,) * (extra + 1))
    margin = margin.masked_fill(~serve.any(-1).view((-1,) + (1,) * (extra + 1)), -1e3)
    return (margin >= 0).to(gamma_lin.dtype) if softness_db is None else torch.sigmoid(margin / softness_db)


def served_rule_weights(reach, served, budget: int):
    """Weight of each K = 1..K_max in the served group's decision: the first K that reaches, kept only when its usage
    served * K is within the budget (otherwise the window is skipped).  reach [B, (R,) K_max] (0/1 or soft), served [B]
    bin counts; returns [B, (R,) K_max], one-hot or zero for hard reach."""
    before = torch.cumprod(torch.cat((torch.ones_like(reach[..., :1]), 1.0 - reach[..., :-1]), dim=-1), dim=-1)
    K = torch.arange(1, LINK_K_MAX + 1, device=reach.device, dtype=reach.dtype)
    within = (served.to(reach.dtype)[:, None] * K[None, :] <= budget)
    return reach * before * within.view((reach.shape[0],) + (1,) * (reach.dim() - 2) + (LINK_K_MAX,))


def served_rule_k(reach, served, budget: int):
    """Return the first reaching K, or zero when no K reaches or its usage exceeds the budget.

    K=0 means the complete window is skipped.
    """
    K = torch.arange(1, LINK_K_MAX + 1, device=reach.device, dtype=reach.dtype)
    return (served_rule_weights(reach, served, budget) * K).sum(-1).round().long()


def apply_repetition_increment(K, served, budget: int, increment: int = 0) -> np.ndarray:
    """Apply the deployment repetition increment without violating K_max or a per-window budget.

    ``K`` and ``served`` are equal-length integer vectors.  K=0 and empty served sets remain skipped.  If an input
    action already exceeds ``budget``, it is skipped rather than reduced to a repetition count that did not satisfy
    the reliability rule.  Every other row returns ``min(K + increment, LINK_K_MAX, floor(budget / served))``.
    ``increment`` is restricted to the evaluator's validation-search range 0..7.
    """
    raw_K, raw_served = np.asarray(K), np.asarray(served)
    if raw_K.ndim != 1 or raw_served.shape != raw_K.shape:
        raise ValueError("K and served must be equal-length vectors")
    if not np.issubdtype(raw_K.dtype, np.integer) or not np.issubdtype(raw_served.dtype, np.integer):
        raise ValueError("K and served must be integer vectors")
    if (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or int(budget) < 1
            or isinstance(increment, (bool, np.bool_)) or not isinstance(increment, (int, np.integer))
            or not 0 <= int(increment) <= INCREMENT_MAX):
        raise ValueError("budget must be a positive integer and increment an integer in 0..7")
    K_array, served_array = raw_K.astype(np.int64, copy=False), raw_served.astype(np.int64, copy=False)
    if np.any((K_array < 0) | (K_array > LINK_K_MAX)) or np.any(served_array < 0):
        raise ValueError("K must be in 0..8 and served nonnegative")
    budget_limit = np.zeros_like(served_array)
    np.floor_divide(int(budget), served_array, out=budget_limit, where=served_array > 0)
    sent = (served_array > 0) & (K_array > 0)
    feasible = sent & (K_array <= budget_limit)
    requested = np.minimum(K_array + int(increment), LINK_K_MAX)
    return np.where(feasible, np.minimum(requested, budget_limit), 0).astype(np.int64, copy=False)




@torch.no_grad()
def decide(net, ck, obs, rows, windows, device="cpu", *, p0_offset_db=0.0, budget=None, increment=0,
           return_base_k=False, chunk=512):
    """Run a checkpoint on episode--window pairs using causal observations only.

    ``rows`` index the leading episode axis of ``obs`` and ``windows`` are 0..11. The return is Boolean ``serve [P, M*N]``, final integer ``K [P]``, and ``level [P, 2]`` for the median and 0.2-quantile forecast in dB. The Chase surrogate multiplies the latest fed-back linear SINR, shifted by the lower forecast minus the checkpoint's validation-selected offset, by K before EESM. The base rule chooses the first K that reaches the model threshold within ``budget``; when omitted, ``budget`` is read from the checkpoint. ``increment`` then applies :func:`apply_repetition_increment`. K=0 skips the complete window, even if ``serve`` contains candidate bins. With ``return_base_k=True``, the base K is appended as a fourth return value.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    windows = np.asarray(windows, dtype=np.int64).reshape(-1)
    Z = torch.as_tensor(network_input(obs, p0_offset_db), device=device)
    if Z.shape[-1] + 1 != ck["cfg"]["C_dd"]:
        raise ValueError(f"network input has {Z.shape[-1] + 1} channels; the checkpoint expects {ck['cfg']['C_dd']}")
    geometry = navigation_geometry(obs)
    mean, std = (np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v, dtype=np.float32)
                 for v in (ck["nav_mean"], ck["nav_std"]))
    protocol_mu = np.asarray(obs["mu_p"], dtype=np.int64)
    threshold = torch.as_tensor(reliability_threshold_db(protocol_mu), dtype=torch.float32, device=device)
    beta = torch.as_tensor(eesm_beta(protocol_mu), dtype=torch.float32, device=device)
    checkpoint_budget = int(ck["budget_provenance"]["budget"])
    decision_budget = checkpoint_budget if budget is None else budget
    if (isinstance(decision_budget, (bool, np.bool_)) or not isinstance(decision_budget, (int, np.integer))
            or int(decision_budget) < 1):
        raise ValueError("budget must be a positive integer")
    decision_budget, offset = int(decision_budget), float(ck["rule_offset_db"])
    serve = np.empty((len(rows), Z.shape[1] * Z.shape[2]), dtype=bool)
    K = np.empty(len(rows), dtype=np.int64)
    level_out = np.empty((len(rows), 2), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        r, w = rows[s:s + chunk], windows[s:s + chunk]
        rt = torch.as_tensor(r, device=device)
        x = with_window_plane(Z[rt], w)
        nav = np.concatenate((np.asarray(obs["z_nav"], dtype=np.float32)[r], geometry[r, w]), axis=1)
        skip_logit, level = net(x, torch.as_tensor((nav - mean) / std, device=device))
        keep = skip_logit.flatten(1) < 0
        gamma = torch.expm1(x[..., 1]).clamp_min(0.0).flatten(1)
        reach = served_rule_reach(gamma, level[:, 1] - offset, threshold[rt], keep, beta[rt])
        serve[s:s + len(r)] = keep.cpu().numpy()
        K[s:s + len(r)] = served_rule_k(reach, keep.sum(-1), decision_budget).cpu().numpy()
        level_out[s:s + len(r)] = level.cpu().numpy()
    base_K = K.copy()
    K = apply_repetition_increment(base_K, serve.sum(1), decision_budget, increment)
    return (serve, K, level_out, base_K) if return_base_k else (serve, K, level_out)


def served_bler_curve(horizon_db, serve, protocol_mu) -> np.ndarray:
    """Evaluate candidate decisions against the stored future horizon; this is not a scheduler input.

    EESM combines the served bins' Chase-accumulated linear SINR at each K and maps it through the model BLER curve. Inputs are ``horizon_db [P, bins, K_max]`` in stored float16 dB, Boolean ``serve [P, bins]``, and ``protocol_mu [P]``. The return is ``[P, K_max]`` and is one for a row with no served bins.
    """
    stored = np.asarray(horizon_db)
    keep = np.asarray(serve, dtype=bool)
    mu = np.asarray(protocol_mu, dtype=np.int64).reshape(-1)
    if stored.ndim != 3 or stored.shape[-1] != LINK_K_MAX or keep.shape != stored.shape[:2] or mu.shape != (len(stored),):
        raise ValueError("served_bler_curve needs horizon_db [P, bins, 8], serve [P, bins] and protocol_mu [P]")
    if stored.dtype != np.dtype(np.float16) or not np.isfinite(stored).all() or np.any((mu < 0) | (mu >= NUM_MCS)):
        raise ValueError("horizon_db must be finite stored float16 and protocol_mu a 0..22 MCS index")
    combined = torch.from_numpy(np.power(10.0, stored.astype(np.float64) / 10.0)).cumsum(-1)        # [P, bins, K]
    beta = torch.as_tensor(_BETA[mu], dtype=torch.float64)[:, None, None]
    member = torch.from_numpy(keep)[..., None]
    lse = torch.logsumexp((-combined / beta).masked_fill(~member, -torch.inf), dim=1)               # [P, K]
    count = member[..., 0].sum(-1).clamp_min(1).to(torch.float64)[:, None]
    eff_db = 10.0 * torch.log10((-beta[:, 0] * (lse - torch.log(count))).clamp_min(1e-20))
    offset = torch.as_tensor(_A[mu], dtype=torch.float64)[:, None] * (eff_db - torch.as_tensor(_B[mu])[:, None])
    bler = torch.exp(-torch.logaddexp(torch.zeros_like(offset), offset)).numpy()
    return np.where(keep.any(1)[:, None], bler, 1.0)


def load_future(root, ids, windows):
    """Load evaluator-only future data from targets.h5 for episode--window pairs.

    Returns the clean horizon ``[P, bins, 8]`` in stored float16 dB and the per-bin repetition map ``[P, bins]``.
    """
    import h5py
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    windows = np.asarray(windows, dtype=np.int64).reshape(-1)
    with h5py.File(Path(root) / "targets.h5", "r") as f:
        horizon = np.stack([f["horizon_db"][int(e), 0, int(w)] for e, w in zip(ids, windows)])
        k_map = np.stack([f["K_map"][int(e), 0, int(w)] for e, w in zip(ids, windows)])
    return horizon.reshape(len(ids), -1, LINK_K_MAX), k_map.reshape(len(ids), -1)


def load_budget_labels(root, ids, windows, budget: int):
    """Load evaluator labels for one budget from budget_labels.h5.

    The returned dictionary has ``feasible [P]``, future-map ``classes [P, bins]``, class factors ``K [P, 4]``, per-bin actions ``K_bin [P, bins]``, ``usage [P]``, and ``utility [P]`` for the selected episode--window pairs. K=0 denotes an empty, skipped, or infeasible label class. ``usage`` is -1 and ``utility`` is NaN when infeasible. These labels are not inputs to ``decide``.
    """
    import h5py
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    windows = np.asarray(windows, dtype=np.int64).reshape(-1)
    with h5py.File(Path(root) / "budget_labels.h5", "r") as f:
        g = f[f"budgets/{int(budget)}"]
        classes = np.stack([f["grouping/mask"][int(e), int(w)] for e, w in zip(ids, windows)]).reshape(len(ids), -1).astype(np.int64)
        K = np.stack([g["K"][int(e), int(w)] for e, w in zip(ids, windows)]).astype(np.int64)
        return {"feasible": np.asarray([g["feasible"][int(e), int(w)] for e, w in zip(ids, windows)], dtype=bool),
                "classes": classes,
                "K": K,
                "K_bin": np.take_along_axis(K, classes, axis=1),
                "usage": np.asarray([g["usage"][int(e), int(w)] for e, w in zip(ids, windows)], dtype=np.int64),
                "utility": np.asarray([g["utility"][int(e), int(w)] for e, w in zip(ids, windows)], dtype=np.float64)}
