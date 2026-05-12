from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.optim as optim

from model import STHurdleGammaBaseline, STHurdleLogNormalBaseline
from model import hurdle_gamma_nll_loss, hurdle_lognormal_nll_loss
from utils import (
    GraphDataset,
    build_bundle,
    hurdle_gamma_interval_from_samples,
    hurdle_gamma_mean,
    hurdle_lognormal_interval_from_samples,
    hurdle_lognormal_mean,
    interval_score,
    mpiw,
    ndcg_at_k,
    picp,
    set_seed,
)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _init_metric_state(horizon: int) -> dict:
    return {
        "mae_all_sum": 0.0, "mae_all_count": 0.0,
        "rmse_all_sum": 0.0, "rmse_all_count": 0.0,
        "mae_pos_sum": 0.0, "mae_pos_count": 0.0,
        "rmse_pos_sum": 0.0, "rmse_pos_count": 0.0,
        "picp_all_sum": 0.0, "picp_all_count": 0.0,
        "mpiw_all_sum": 0.0, "mpiw_all_count": 0.0,
        "is_all_sum": 0.0, "is_all_count": 0.0,
        "recall_hits": np.zeros(horizon, dtype=np.float64),
        "recall_denoms": np.zeros(horizon, dtype=np.float64),
        "ndcg_sums": np.zeros(horizon, dtype=np.float64),
        "ndcg_counts": np.zeros(horizon, dtype=np.float64),
    }


def _update_metrics(
        state: dict,
        y_true: np.ndarray,
        y_rank: np.ndarray,
        pred_mean: np.ndarray,
        hr_ratio: float,
        interval_alpha: float,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
):
    count = y_true.size
    err = y_true - pred_mean
    state["mae_all_sum"] += float(np.sum(np.abs(err)))
    state["mae_all_count"] += count
    state["rmse_all_sum"] += float(np.sum(err ** 2))
    state["rmse_all_count"] += count

    pos_mask = y_true > 0
    pos_count = int(np.sum(pos_mask))
    if pos_count > 0:
        err_pos = err[pos_mask]
        state["mae_pos_sum"] += float(np.sum(np.abs(err_pos)))
        state["mae_pos_count"] += pos_count
        state["rmse_pos_sum"] += float(np.sum(err_pos ** 2))
        state["rmse_pos_count"] += pos_count

    horizon = y_rank.shape[-1]
    for t in range(horizon):
        y_t = y_rank[:, :, t]
        score_t = pred_mean[:, :, t]
        y_flat = np.asarray(y_t).reshape(-1)
        score_flat = np.asarray(score_t).reshape(-1)
        k = max(1, int(np.ceil(float(hr_ratio) * score_flat.shape[0])))
        order = np.argsort(-score_flat, kind="mergesort")[:k]
        event = y_flat > 0
        total_pos = int(np.sum(event))
        if total_pos > 0:
            hits = int(np.sum(event[order]))
            state["recall_hits"][t] += hits
            state["recall_denoms"][t] += total_pos
        ndcg_val = ndcg_at_k(y_t, score_t, top_frac=hr_ratio)
        if not np.isnan(ndcg_val):
            state["ndcg_sums"][t] += ndcg_val
            state["ndcg_counts"][t] += 1

    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        mpiw_val = mpiw(lower, upper)
        state["mpiw_all_sum"] += mpiw_val * count
        state["mpiw_all_count"] += count

        picp_val = picp(y_true, lower, upper)
        state["picp_all_sum"] += picp_val * count
        state["picp_all_count"] += count

        is_val = interval_score(y_true, lower, upper, alpha=interval_alpha)
        state["is_all_sum"] += is_val * count
        state["is_all_count"] += count


def _finalize_metrics(state: dict) -> dict:
    metrics = {}
    for key in ("mae_all", "rmse_all", "mae_pos", "rmse_pos", "picp_all", "mpiw_all", "is_all"):
        s = state.get(f"{key}_sum", 0.0)
        c = state.get(f"{key}_count", 0.0)
        metrics[key] = float(s / c) if c > 0 else float("nan")

    metrics["rmse_all"] = float(np.sqrt(
        state["rmse_all_sum"] / state["rmse_all_count"]
    )) if state["rmse_all_count"] > 0 else float("nan")
    metrics["rmse_pos"] = float(np.sqrt(
        state["rmse_pos_sum"] / state["rmse_pos_count"]
    )) if state["rmse_pos_count"] > 0 else float("nan")

    if state["recall_hits"] is not None:
        recall_t = np.divide(
            state["recall_hits"], state["recall_denoms"],
            out=np.full_like(state["recall_hits"], np.nan),
            where=state["recall_denoms"] > 0,
        )
        metrics["recall"] = float(np.nanmean(recall_t))

    if state["ndcg_sums"] is not None:
        ndcg_t = np.divide(
            state["ndcg_sums"], state["ndcg_counts"],
            out=np.full_like(state["ndcg_sums"], np.nan),
            where=state["ndcg_counts"] > 0,
        )
        metrics["ndcg"] = float(np.nanmean(ndcg_t))
    return metrics


def _merge_spaces(target_metrics: dict, event_metrics: dict) -> dict:
    merged = dict(target_metrics)
    merged["ts_recall"] = float(target_metrics.get("recall", float("nan")))
    merged["es_recall"] = float(event_metrics.get("recall", float("nan")))
    merged["ts_ndcg"] = float(target_metrics.get("ndcg", float("nan")))
    merged["es_ndcg"] = float(event_metrics.get("ndcg", float("nan")))
    return merged


def _format_metrics(prefix: str, metrics: dict) -> str:
    order = [
        ("MAE_all", "mae_all"), ("RMSE_all", "rmse_all"),
        ("MAE_pos", "mae_pos"), ("RMSE_pos", "rmse_pos"),
        ("TS_Recall@20%", "ts_recall"), ("ES_Recall@20%", "es_recall"),
        ("TS_NDCG@20%", "ts_ndcg"), ("ES_NDCG@20%", "es_ndcg"),
        ("PICP_all", "picp_all"), ("MPIW_all", "mpiw_all"),
        ("IS_all", "is_all"),
    ]
    parts = [prefix]
    for label, key in order:
        v = metrics.get(key, float("nan"))
        parts.append(f"{label}={'N/A' if not np.isfinite(v) else f'{v:.4f}'}")
    return " | ".join(parts)


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        loader,
        bundle: dict,
        device: torch.device,
        hr_ratio: float,
        interval_alpha: float,
        model_kind: str,
        interval_samples: int,
        compute_intervals: bool = True,
) -> dict:
    target_state = _init_metric_state(bundle["horizon"])
    event_state = _init_metric_state(bundle["horizon"])
    model.eval()

    for batch in loader:
        batch = _move_batch(batch, device)
        x = batch["x"]
        y_np = batch["y"].cpu().numpy()
        y_rank_np = batch["y_rank"].cpu().numpy()
        y_event_np = batch["y_event"].cpu().numpy()
        y_event_rank_np = batch["y_event_rank"].cpu().numpy()

        q_logits, p2, p3 = model(x, bundle["A_q"], bundle["A_h"])

        if model_kind == "hurdle_gamma":
            pred_mean = hurdle_gamma_mean(q_logits, p2).cpu().numpy()
        else:
            pred_mean = hurdle_lognormal_mean(q_logits, p2, p3).cpu().numpy()

        lower = upper = None
        if compute_intervals:
            if model_kind == "hurdle_gamma":
                lower, upper = hurdle_gamma_interval_from_samples(
                    q_logits, p2, p3, alpha=interval_alpha, num_samples=interval_samples,
                )
            else:
                lower, upper = hurdle_lognormal_interval_from_samples(
                    q_logits, p2, p3, alpha=interval_alpha, num_samples=interval_samples,
                )

        for state, y, y_r in [
            (target_state, y_np, y_rank_np),
            (event_state, y_event_np, y_event_rank_np),
        ]:
            _update_metrics(state, y, y_r, pred_mean, hr_ratio, interval_alpha, lower, upper)

    return _merge_spaces(_finalize_metrics(target_state), _finalize_metrics(event_state))


def train_one_epoch(
        model: torch.nn.Module,
        loader,
        bundle: dict,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        model_kind: str,
        clip_grad: float,
        hurdle_pos_weight: float = 0.0,
        hurdle_event_occ_lambda: float = 0.0,
        hurdle_event_occ_weight_cap: float = 20.0,
) -> float:
    model.train()
    losses = []
    pos_weight = None
    if model_kind in ("hurdle_gamma", "hurdle_lognormal") and hurdle_pos_weight > 0:
        pos_weight = torch.tensor(float(hurdle_pos_weight), device=device)

    for batch in loader:
        batch = _move_batch(batch, device)
        x = batch["x"]
        y = batch["y"]
        z_occ = (batch["y_event"] > 0).float()

        event_pos_weight = None
        if hurdle_event_occ_lambda > 0:
            pc = bundle["train_event_pos_count"]
            nc = bundle["train_event_neg_count"]
            if pc > 0:
                w = min(nc / float(pc), float(hurdle_event_occ_weight_cap))
                event_pos_weight = torch.tensor(w, device=device)

        optimizer.zero_grad(set_to_none=True)
        q_logits, p2, p3 = model(x, bundle["A_q"], bundle["A_h"])

        if model_kind == "hurdle_gamma":
            loss_dict = hurdle_gamma_nll_loss(
                y, q_logits, p2, p3,
                z_occ=z_occ,
                event_occ_lambda=hurdle_event_occ_lambda,
                pos_weight=pos_weight,
                event_pos_weight=event_pos_weight,
            )
        else:
            loss_dict = hurdle_lognormal_nll_loss(
                y, q_logits, p2, p3,
                z_occ=z_occ,
                event_occ_lambda=hurdle_event_occ_lambda,
                pos_weight=pos_weight,
                event_pos_weight=event_pos_weight,
            )

        loss = loss_dict["loss"]
        if not torch.isfinite(loss).item():
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


def train(
        model: torch.nn.Module,
        bundle: dict,
        args,
        device: torch.device,
        model_kind: str,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("-inf")
    best_state = None
    best_val_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, bundle["train_loader"], bundle, device, optimizer,
            model_kind=model_kind,
            clip_grad=args.clip_grad,
            hurdle_pos_weight=args.hurdle_occ_pos_weight,
            hurdle_event_occ_lambda=args.hurdle_event_occ_lambda,
            hurdle_event_occ_weight_cap=args.hurdle_event_occ_pos_weight_cap,
        )

        val_metrics = evaluate(
            model, bundle["val_loader"], bundle, device,
            hr_ratio=args.hr_ratio,
            interval_alpha=args.interval_alpha,
            model_kind=model_kind,
            interval_samples=args.mc_interval_samples,
            compute_intervals=False,
        )
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | {_format_metrics('val', val_metrics)}")

        if val_metrics["es_recall"] > best_val:
            best_val = val_metrics["es_recall"]
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = dict(val_metrics)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, bundle["test_loader"], bundle, device,
        hr_ratio=args.hr_ratio,
        interval_alpha=args.interval_alpha,
        model_kind=model_kind,
        interval_samples=args.mc_interval_samples,
        compute_intervals=True,
    )
    print(_format_metrics("test", test_metrics))

    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return {
        "val_metrics": best_val_metrics if best_val_metrics is not None else val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": ckpt_path,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ST-Hurdle road risk forecasting")
    p.add_argument("--model", type=str, default="st_hurdle_lognormal",
                   choices=["st_hurdle_gamma", "st_hurdle_lognormal"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--history_len", type=int, default=14)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--start_date", type=str, default="2021-01-01")
    p.add_argument("--end_date", type=str, default="2024-12-31")
    p.add_argument("--diffuse_alpha", type=float, default=0.3)
    p.add_argument("--diffuse_k", type=int, default=2)
    p.add_argument("--diffuse_weight", type=float, default=0.5)
    p.add_argument("--diffuse_norm", type=str, default="none", choices=["row", "none"])
    p.add_argument("--risk_diffuse_zero_thresh", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--hr_ratio", type=float, default=0.2)
    p.add_argument("--interval_alpha", type=float, default=0.05)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--temp_layers", type=int, default=3)
    p.add_argument("--spatial_layers", type=int, default=3)
    p.add_argument("--diffusion_order", type=int, default=2)
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--hurdle_max_mu", type=float, default=20.0)
    p.add_argument("--hurdle_gamma_max_alpha", type=float, default=50.0)
    p.add_argument("--hurdle_lognormal_max_sigma", type=float, default=3.0)
    p.add_argument("--hurdle_occ_pos_weight", type=float, default=0)
    p.add_argument("--hurdle_event_occ_lambda", type=float, default=0.3)
    p.add_argument("--hurdle_event_occ_pos_weight_cap", type=float, default=20.0)
    p.add_argument("--mc_interval_samples", type=int, default=100)
    p.add_argument("--max_nodes", type=int, default=0, help="Max nodes to use (0 = all). Use for quick testing.")
    return p


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    bundle = build_bundle(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        history_len=args.history_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        diffuse_alpha=args.diffuse_alpha,
        diffuse_k=args.diffuse_k,
        diffuse_weight=args.diffuse_weight,
        diffuse_norm=args.diffuse_norm,
        risk_diffuse_zero_thresh=args.risk_diffuse_zero_thresh,
        device=device,
        max_nodes=args.max_nodes,
    )
    bundle["horizon"] = args.horizon
    print(f"Nodes: {bundle['num_nodes']} | Input dim: {bundle['graph_input_dim']}")

    if args.model == "st_hurdle_gamma":
        model = STHurdleGammaBaseline(
            input_dim=bundle["graph_input_dim"],
            hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            kernel_size=args.kernel_size,
            temp_layers=args.temp_layers,
            spatial_layers=args.spatial_layers,
            diffusion_order=args.diffusion_order,
            dropout=args.dropout,
            max_mu=args.hurdle_max_mu,
            max_alpha=args.hurdle_gamma_max_alpha,
        ).to(device=device)
        model_kind = "hurdle_gamma"
    else:
        model = STHurdleLogNormalBaseline(
            input_dim=bundle["graph_input_dim"],
            hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            kernel_size=args.kernel_size,
            temp_layers=args.temp_layers,
            spatial_layers=args.spatial_layers,
            diffusion_order=args.diffusion_order,
            dropout=args.dropout,
            max_sigma=args.hurdle_lognormal_max_sigma,
        ).to(device=device)
        model_kind = "hurdle_lognormal"

    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    result = train(model, bundle, args, device, model_kind)

    summary_path = os.path.join(args.out_dir, f"{args.model}_summary.json")
    result["model"] = args.model
    result["args"] = vars(args)

    def _sanitize(v):
        if isinstance(v, dict):
            return {k: _sanitize(vv) for k, vv in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        if isinstance(v, list):
            return [_sanitize(vv) for vv in v]
        return v

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(result), f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
