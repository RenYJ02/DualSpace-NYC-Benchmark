from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
LOGNORMAL_EXP_CLAMP = 20.0

def log_lognormal_pdf(
        y: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    y = y.clamp_min(eps)
    sigma_log = sigma_log.clamp_min(eps)
    log_y = torch.log(y)
    z = (log_y - mu_log) / sigma_log
    return (
            -torch.log(y)
            - torch.log(sigma_log)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * z * z
    )

def log_gamma_pdf_mu_alpha(
        y: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    y = y.clamp_min(eps)
    mu = mu.clamp_min(eps)
    alpha = alpha.clamp_min(eps)
    return (
            alpha * (torch.log(alpha) - torch.log(mu))
            - torch.lgamma(alpha)
            + (alpha - 1.0) * torch.log(y)
            - alpha * y / mu
    )

def hurdle_lognormal_nll_loss(
        y: torch.Tensor,
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        z_occ: torch.Tensor | None = None,
        event_occ_lambda: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        event_pos_weight: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    y = y.float().clamp_min(0.0)
    z = (y > 0).float()

    occ_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        occ_kwargs["pos_weight"] = pos_weight
    occ_nll = F.binary_cross_entropy_with_logits(q_logits, z, **occ_kwargs)

    safe_y = torch.where(z > 0, y, torch.ones_like(y))
    log_pdf = log_lognormal_pdf(safe_y, mu_log=mu_log, sigma_log=sigma_log, eps=eps)
    pos_nll = -log_pdf
    total = occ_nll + z * pos_nll
    aux_occ = torch.zeros_like(total)
    if z_occ is not None and event_occ_lambda > 0.0:
        event_kwargs = {"reduction": "none"}
        if event_pos_weight is not None:
            event_kwargs["pos_weight"] = event_pos_weight
        aux_occ = F.binary_cross_entropy_with_logits(q_logits, z_occ.float(), **event_kwargs)
        total = total + event_occ_lambda * aux_occ

    if reduction == "sum":
        return {
            "loss": total.sum(),
            "occ_nll": occ_nll.sum(),
            "pos_nll": (z * pos_nll).sum(),
            "aux_occ": aux_occ.sum() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                       device=y.device),
        }
    if reduction == "none":
        return {
            "loss": total,
            "occ_nll": occ_nll,
            "pos_nll": z * pos_nll,
            "aux_occ": aux_occ,
        }
    return {
        "loss": total.mean(),
        "occ_nll": occ_nll.mean(),
        "pos_nll": (z * pos_nll).sum() / z.sum().clamp_min(1.0),
        "aux_occ": aux_occ.mean() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                    device=y.device),
    }


def hurdle_gamma_nll_loss(
        y: torch.Tensor,
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        z_occ: torch.Tensor | None = None,
        event_occ_lambda: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        event_pos_weight: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    y = y.float().clamp_min(0.0)
    z = (y > 0).float()

    occ_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        occ_kwargs["pos_weight"] = pos_weight
    occ_nll = F.binary_cross_entropy_with_logits(q_logits, z, **occ_kwargs)

    safe_y = torch.where(z > 0, y, torch.ones_like(y))
    log_pdf = log_gamma_pdf_mu_alpha(safe_y, mu=mu, alpha=alpha, eps=eps)
    pos_nll = -log_pdf
    total = occ_nll + z * pos_nll
    aux_occ = torch.zeros_like(total)
    if z_occ is not None and event_occ_lambda > 0.0:
        event_kwargs = {"reduction": "none"}
        if event_pos_weight is not None:
            event_kwargs["pos_weight"] = event_pos_weight
        aux_occ = F.binary_cross_entropy_with_logits(q_logits, z_occ.float(), **event_kwargs)
        total = total + event_occ_lambda * aux_occ

    if reduction == "sum":
        return {
            "loss": total.sum(),
            "occ_nll": occ_nll.sum(),
            "pos_nll": (z * pos_nll).sum(),
            "aux_occ": aux_occ.sum() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                       device=y.device),
        }
    if reduction == "none":
        return {
            "loss": total,
            "occ_nll": occ_nll,
            "pos_nll": z * pos_nll,
            "aux_occ": aux_occ,
        }
    return {
        "loss": total.mean(),
        "occ_nll": occ_nll.mean(),
        "pos_nll": (z * pos_nll).sum() / z.sum().clamp_min(1.0),
        "aux_occ": aux_occ.mean() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                    device=y.device),
    }


def _support_mm(support: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if support.is_sparse:
        support = support.coalesce()
        return torch.stack([torch.sparse.mm(support, x_i) for x_i in x], dim=0)
    return torch.einsum("nm,bmf->bnf", support, x)


class CausalTemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filter_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.kernel_size))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.kernel_size))
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        filt = self.filter_conv(x_pad)
        gate = torch.sigmoid(self.gate_conv(x_pad))
        y = filt * gate + self.residual(x)
        return self.dropout(F.relu(y))


class DiffusionConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, order: int, num_supports: int = 1):
        super().__init__()
        self.order = int(order)
        self.num_supports = int(num_supports)
        self.linear = nn.Linear(in_dim * (1 + self.order * self.num_supports), out_dim)

    def forward(self, x: torch.Tensor, supports: Iterable[torch.Tensor]) -> torch.Tensor:
        x_terms: List[torch.Tensor] = [x]
        for support in supports:
            x0 = x
            x1 = _support_mm(support, x0)
            x_terms.append(x1)
            prev_prev, prev = x0, x1
            for _ in range(2, self.order + 1):
                current = 2.0 * _support_mm(support, prev) - prev_prev
                x_terms.append(current)
                prev_prev, prev = prev, current
        return self.linear(torch.cat(x_terms, dim=-1))


class HurdleGammaHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, horizon)
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.alpha_proj = nn.Linear(hidden_dim, horizon)
        self.max_mu = float(max_mu)
        self.max_alpha = float(max_alpha)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_logits = self.q_proj(h)
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.max_mu)
        alpha = (F.softplus(self.alpha_proj(h)) + self.eps).clamp(self.eps, self.max_alpha)
        return q_logits, mu, alpha


class HurdleLogNormalHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_sigma: float = 2.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, horizon)
        self.mu_log_proj = nn.Linear(hidden_dim, horizon)
        self.sigma_log_proj = nn.Linear(hidden_dim, horizon)
        self.max_sigma = float(max_sigma)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_logits = self.q_proj(h)
        mu_log = self.mu_log_proj(h)
        sigma_log = (F.softplus(self.sigma_log_proj(h)) + self.eps).clamp(self.eps, self.max_sigma)
        return q_logits, mu_log, sigma_log


class SharedHurdleEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            temp_layers: int,
            spatial_layers: int,
            diffusion_order: int,
            dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.temporal_blocks = nn.ModuleList(
            [CausalTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout) for _ in range(temp_layers)]
        )
        self.spatial_blocks = nn.ModuleList(
            [DiffusionConv(hidden_dim, hidden_dim, diffusion_order, num_supports=2) for _ in range(spatial_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, history_len, feat_dim = x.shape
        h = x.permute(0, 3, 1, 2).contiguous()
        h = self.input_proj(h)
        for block in self.temporal_blocks:
            h = block(h)
        h = h.permute(0, 3, 2, 1).contiguous()

        supports = [A_q, A_h]
        for gconv in self.spatial_blocks:
            h_bt = h.reshape(batch_size * history_len, h.size(2), self.hidden_dim)
            h_bt = self.dropout(F.relu(gconv(h_bt, supports)))
            h = h_bt.view(batch_size, history_len, h.size(2), self.hidden_dim)

        h_last = h[:, -1, :, :]
        h_mean = h.mean(dim=1)
        return self.norm(0.5 * (h_last + h_mean))


class STHurdleGammaBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
        )
        self.head = HurdleGammaHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_mu=max_mu,
            max_alpha=max_alpha,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STHurdleLogNormalBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            max_sigma: float = 2.0,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
        )
        self.head = HurdleLogNormalHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_sigma=max_sigma,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)
