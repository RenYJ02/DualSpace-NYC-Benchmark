from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

EPS = 1e-6
LOGNORMAL_EXP_CLAMP = 20.0


# =============================================================================
# Graph operations
# =============================================================================

def get_normalized_adj(A):
    if sp.issparse(A):
        diag = A.diagonal()
        if np.all(diag == 0):
            A = A + sp.eye(A.shape[0], dtype=np.float32, format="coo")
        D = np.array(A.sum(axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5
        d_inv_sqrt = np.reciprocal(np.sqrt(D))
        d_mat = sp.diags(d_inv_sqrt)
        return (d_mat.dot(A).dot(d_mat)).tocoo()
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    if sp.issparse(adj_mx):
        adj_mx = adj_mx.tocoo()
        d = np.array(adj_mx.sum(1)).reshape((-1,))
        d_inv = np.power(d, -1, where=d > 0)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
        return random_walk_mx
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def scipy_to_torch_sparse(mat: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = mat.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data.astype(np.float32)
    i = torch.from_numpy(indices).long()
    v = torch.from_numpy(values).float()
    return torch.sparse_coo_tensor(i, v, coo.shape, device=device).coalesce()


def make_undirected_edge_index(edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    rev = np.stack([dst, src], axis=0)
    return np.concatenate([edge_index, rev], axis=1)


def apply_mask_to_edge_index(edge_index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.all():
        return edge_index
    node_idx = np.where(mask)[0]
    mapper = -np.ones(mask.shape[0], dtype=np.int64)
    mapper[node_idx] = np.arange(node_idx.shape[0])
    src, dst = edge_index
    keep = mask[src] & mask[dst]
    src_new = mapper[src[keep]]
    dst_new = mapper[dst[keep]]
    return np.stack([src_new, dst_new], axis=0)


def compute_structural_node_features(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    if num_nodes <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if edge_index.size == 0:
        return np.zeros((num_nodes, 2), dtype=np.float32)

    src, dst = edge_index
    A = sp.coo_matrix(
        (np.ones(src.shape[0], dtype=np.float32), (src, dst)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0

    degree = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float32)
    degree_centrality = degree / float(max(num_nodes - 1, 1))

    pr = np.full(num_nodes, 1.0 / max(num_nodes, 1), dtype=np.float64)
    degree64 = degree.astype(np.float64)
    inv_degree = np.zeros_like(degree64)
    nonzero = degree64 > 0
    inv_degree[nonzero] = 1.0 / degree64[nonzero]
    P = sp.diags(inv_degree).dot(A).tocsr().astype(np.float64)
    dangling = ~nonzero
    alpha = 0.85
    teleport = (1.0 - alpha) / max(num_nodes, 1)
    for _ in range(100):
        dangling_mass = pr[dangling].sum()
        pr_next = alpha * (P.T @ pr + dangling_mass / max(num_nodes, 1)) + teleport
        if np.abs(pr_next - pr).sum() < 1e-8:
            pr = pr_next
            break
        pr = pr_next

    features = np.stack([degree_centrality, pr.astype(np.float32)], axis=1)
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)
    return ((features - feat_mean) / feat_std).astype(np.float32)


def build_diffusion_supports(
        edge_index_undirected: np.ndarray,
        num_nodes: int,
        device: torch.device,
):
    A = sp.coo_matrix(
        (
            np.ones(edge_index_undirected.shape[1], dtype=np.float32),
            (edge_index_undirected[0], edge_index_undirected[1]),
        ),
        shape=(num_nodes, num_nodes),
    )
    A_wave = get_normalized_adj(A)
    A_q = calculate_random_walk_matrix(A_wave).T
    A_h = calculate_random_walk_matrix(A_wave.T).T

    if sp.issparse(A_q):
        A_q = scipy_to_torch_sparse(A_q, device)
        A_h = scipy_to_torch_sparse(A_h, device)
    else:
        A_q = torch.from_numpy(A_q.astype("float32")).to(device=device)
        A_h = torch.from_numpy(A_h.astype("float32")).to(device=device)
    return A_q, A_h


# =============================================================================
# Seed and date utilities
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_date_arg(s: str) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s)


# =============================================================================
# Time splits
# =============================================================================

def build_time_splits(T: int, history_len: int, horizon: int, train_ratio: float = 8 / 12, val_ratio: float = 2 / 12):
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    max_start = T - (history_len + horizon)
    all_s = list(range(0, max_start + 1))

    train_s, val_s, test_s = [], [], []
    for s in all_s:
        tgt_start = s + history_len
        tgt_end = s + history_len + horizon
        if tgt_end <= train_end:
            train_s.append(s)
        elif (tgt_start >= train_end) and (tgt_end <= val_end):
            val_s.append(s)
        elif tgt_start >= val_end and tgt_end <= T:
            test_s.append(s)
    return train_s, val_s, test_s


# =============================================================================
# Data loading
# =============================================================================

def load_mask(data_dir: str, mask_borough: str, mask_path: str, num_nodes: int) -> np.ndarray:
    if mask_path:
        path = mask_path
    else:
        name = mask_borough.strip().lower()
        if name in ("all", ""):
            return np.ones(num_nodes, dtype=bool)
        borough_map = {
            "bronx": "mask_borough_bronx.npz",
            "brooklyn": "mask_borough_brooklyn.npz",
            "manhattan": "mask_borough_manhattan.npz",
            "queens": "mask_borough_queens.npz",
            "staten island": "mask_borough_staten_island.npz",
            "staten_island": "mask_borough_staten_island.npz",
        }
        if name not in borough_map:
            raise ValueError(f"Unknown mask_borough: {mask_borough}")
        path = os.path.join(data_dir, "line_graph", borough_map[name])

    z = np.load(path)
    key = "mask.npy" if "mask.npy" in z.files else z.files[0]
    mask = z[key].astype(bool)
    if mask.shape[0] != num_nodes:
        raise ValueError(f"Mask size mismatch: {mask.shape[0]} vs {num_nodes}")
    return mask


def load_nyc_zinb_data(
        data_dir: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
):
    node_map_path = os.path.join(data_dir, "line_graph", "node_id_map.parquet")
    links_path = os.path.join(data_dir, "links_with_features.parquet")
    labels_path = os.path.join(data_dir, "crashes_daily_labels.parquet")
    weather_path = os.path.join(data_dir, "weather", "nyc_weather_daily.parquet")
    calendar_path = os.path.join(data_dir, "calendar", "calendar_day.parquet")
    scaler_path = os.path.join(data_dir, "links_features_scaler.json")
    borough_map_path = os.path.join(data_dir, "line_graph", "borough_id_mapping.json")

    node_df = pd.read_parquet(node_map_path).sort_values("node_idx")
    num_nodes = len(node_df)

    links_df = pd.read_parquet(links_path)
    static_cols = [c for c in links_df.columns if c.startswith("x_")]
    if "m_pop_imputed" in links_df.columns:
        static_cols.append("m_pop_imputed")
    static_df = node_df.merge(links_df[["id_geom_hash"] + static_cols], on="id_geom_hash", how="left")
    if static_df[static_cols].isnull().any().any():
        static_df[static_cols] = static_df[static_cols].fillna(0.0)
    X_static = static_df[static_cols].to_numpy(dtype=np.float32)

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    col_to_idx = {c: i for i, c in enumerate(static_cols)}
    for col, stats in scaler.get("features", {}).items():
        if col in col_to_idx:
            idx = col_to_idx[col]
            std = stats.get("std", 1.0)
            if std == 0:
                std = 1.0
            X_static[:, idx] = (X_static[:, idx] - stats.get("mean", 0.0)) / std

    labels_df = pd.read_parquet(labels_path)
    weather_df = pd.read_parquet(weather_path)
    calendar_df = pd.read_parquet(calendar_path)

    for df in (labels_df, weather_df, calendar_df):
        df["id_date"] = pd.to_datetime(df["id_date"])

    min_date = max(calendar_df["id_date"].min(), weather_df["id_date"].min(), labels_df["id_date"].min())
    max_date = min(calendar_df["id_date"].max(), weather_df["id_date"].max(), labels_df["id_date"].max())
    if start_date is not None:
        min_date = max(min_date, start_date)
    if end_date is not None:
        max_date = min(max_date, end_date)
    if min_date > max_date:
        raise ValueError(f"Invalid date range after intersection: {min_date} > {max_date}")

    calendar_df = calendar_df[
        (calendar_df["id_date"] >= min_date) & (calendar_df["id_date"] <= max_date)
        ].sort_values("id_date")
    weather_df = weather_df[(weather_df["id_date"] >= min_date) & (weather_df["id_date"] <= max_date)]
    labels_df = labels_df[(labels_df["id_date"] >= min_date) & (labels_df["id_date"] <= max_date)]

    dates = calendar_df["id_date"].tolist()
    T = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    cal_cols = [c for c in calendar_df.columns if c.startswith("x_cal_")]
    calendar = calendar_df[cal_cols].to_numpy(dtype=np.float32)

    with open(borough_map_path, "r", encoding="utf-8") as f:
        borough_map = json.load(f)["borough_to_id"]
    weather_cols = [c for c in weather_df.columns if c.startswith("x_wth_")]
    weather = np.zeros((len(borough_map), T, len(weather_cols)), dtype=np.float32)
    for borough_name, borough_id in borough_map.items():
        sub = weather_df[weather_df["id_borough"] == borough_name].set_index("id_date")
        sub = sub.reindex(dates)
        if sub[weather_cols].isnull().any().any():
            raise ValueError(f"Weather missing values for {borough_name}")
        weather[borough_id] = sub[weather_cols].to_numpy(dtype=np.float32)

    crash_col = "y_crash_count"
    inj_col = "y_injured_sum"
    kill_col = "y_killed_sum"
    missing_cols = [c for c in (crash_col, inj_col, kill_col) if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns: {missing_cols}. Available: {labels_df.columns.tolist()}")

    node_map = pd.Series(node_df["node_idx"].values, index=node_df["id_geom_hash"]).to_dict()
    labels_df["node_idx"] = labels_df["id_geom_hash"].map(node_map)
    labels_df["date_idx"] = labels_df["id_date"].map(date_to_idx)
    if labels_df["node_idx"].isnull().any() or labels_df["date_idx"].isnull().any():
        raise ValueError("Label rows contain unknown node or date")

    risk = (
            labels_df[crash_col].fillna(0).astype(np.int64)
            + 2 * labels_df[inj_col].fillna(0).astype(np.int64)
            + 3 * labels_df[kill_col].fillna(0).astype(np.int64)
    )
    crash = labels_df[crash_col].fillna(0).astype(np.int64)
    Y_risk = np.zeros((num_nodes, T), dtype=np.float32)
    Y_crash = np.zeros((num_nodes, T), dtype=np.float32)
    Y_risk[labels_df["node_idx"].values, labels_df["date_idx"].values] = risk.to_numpy(dtype=np.float32)
    Y_crash[labels_df["node_idx"].values, labels_df["date_idx"].values] = crash.to_numpy(dtype=np.float32)

    borough_id_arr = np.load(os.path.join(data_dir, "line_graph", "borough_id.npy")).astype(np.int64)
    if borough_id_arr.shape[0] != num_nodes:
        raise ValueError("borough_id.npy size mismatch")

    return X_static, Y_risk, Y_crash, calendar, weather, borough_id_arr, dates


def load_borough_name_order(data_dir: str) -> List[str]:
    mapping_path = os.path.join(data_dir, "line_graph", "borough_id_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        borough_to_id = json.load(f)["borough_to_id"]
    return [name for name, _ in sorted(borough_to_id.items(), key=lambda kv: kv[1])]


# =============================================================================
# Preprocessing
# =============================================================================

def standardize_time_features(
        calendar: np.ndarray,
        weather: np.ndarray,
        train_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cal_mean = calendar[:train_end].mean(axis=0)
    cal_std = calendar[:train_end].std(axis=0)
    cal_std = np.where(cal_std == 0, 1.0, cal_std)
    calendar = (calendar - cal_mean) / cal_std

    w_mean = weather[:, :train_end, :].mean(axis=(0, 1))
    w_std = weather[:, :train_end, :].std(axis=(0, 1))
    w_std = np.where(w_std == 0, 1.0, w_std)
    weather = (weather - w_mean) / w_std
    return calendar, weather


def standardize_node_time_feature(X: np.ndarray, train_end: int) -> Tuple[np.ndarray, float, float]:
    mean = float(X[:, :train_end].mean())
    std = float(X[:, :train_end].std())
    if std == 0.0:
        std = 1.0
    X = (X - mean) / std
    return X.astype(np.float32), mean, std


# =============================================================================
# Prior features
# =============================================================================

def build_exclusive_cumulative_mean(Y: np.ndarray) -> np.ndarray:
    Y = Y.astype(np.float32, copy=False)
    out = np.zeros_like(Y, dtype=np.float32)
    if Y.shape[1] <= 1:
        return out
    cumsum = np.cumsum(Y, axis=1, dtype=np.float32)
    denom = np.arange(1, Y.shape[1], dtype=np.float32)
    out[:, 1:] = cumsum[:, :-1] / denom[None, :]
    return out


def diffuse_risk_time_space(
        Y: np.ndarray,
        edge_index: np.ndarray,
        alpha: float = 0.5,
        k: int = 3,
        hop_weight: float = 0.5,
        undirected: bool = True,
        normalize: str = "row",
        zero_thresh: float = 0.2,
) -> np.ndarray:
    Y = np.asarray(Y)
    edge_index = np.asarray(edge_index)

    if Y.ndim != 2:
        raise ValueError(f"Y must have shape [N, T], got {Y.shape}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if hop_weight < 0:
        raise ValueError(f"hop_weight must be >= 0, got {hop_weight}")
    if normalize not in ("row", "none"):
        raise ValueError("normalize must be 'row' or 'none'")
    if zero_thresh < 0:
        raise ValueError(f"zero_thresh must be >= 0, got {zero_thresh}")

    Y = Y.astype(np.float32, copy=False)
    N, T = Y.shape

    Yt = np.empty((N, T), dtype=np.float32)
    Yt[:, 0] = Y[:, 0]
    for t in range(1, T):
        Yt[:, t] = alpha * Y[:, t] + (1.0 - alpha) * Yt[:, t - 1]

    if k == 0 or hop_weight == 0.0:
        if not np.isfinite(Yt).all():
            raise ValueError("Temporal smoothing produced NaN/Inf values.")
        return Yt

    src0 = edge_index[0].astype(np.int64, copy=False)
    dst0 = edge_index[1].astype(np.int64, copy=False)
    valid = (src0 >= 0) & (src0 < N) & (dst0 >= 0) & (dst0 < N)
    src0 = src0[valid]
    dst0 = dst0[valid]

    if undirected:
        src = np.concatenate([src0, dst0])
        dst = np.concatenate([dst0, src0])
    else:
        src, dst = src0, dst0

    A = sp.coo_matrix(
        (np.ones(src.shape[0], dtype=np.float32), (src, dst)),
        shape=(N, N),
        dtype=np.float32,
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()

    if normalize == "row":
        deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float32)
        inv_deg = np.zeros_like(deg, dtype=np.float32)
        nz = deg > 0
        inv_deg[nz] = 1.0 / deg[nz]
        P = sp.diags(inv_deg).dot(A).tocsr()
    else:
        P = A.tocsr()

    out = Yt.copy()
    walk = Yt.copy()
    total_w = 1.0
    for step in range(1, k + 1):
        walk = P.dot(walk).astype(np.float32, copy=False)
        w = float(hop_weight) ** step
        out += w * walk
        total_w += w
    out /= total_w

    if zero_thresh > 0:
        out = out.copy()
        out[out < float(zero_thresh)] = 0.0

    if not np.isfinite(out).all():
        raise ValueError("Risk diffusion produced NaN/Inf values.")
    return out.astype(np.float32, copy=False)


# =============================================================================
# Task helpers
# =============================================================================

def normalize_target_mode(target_mode: str) -> str:
    mode = str(target_mode).strip().lower()
    if mode not in ("risk", "risk_diffuse"):
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return mode


def normalize_hist_source_mode(hist_source_mode: str) -> str:
    mode = str(hist_source_mode).strip().lower()
    if mode not in ("raw_risk", "target", "both"):
        raise ValueError(f"Unsupported hist_source_mode: {hist_source_mode}")
    return mode


def uses_diffused_target(target_mode: str) -> bool:
    return normalize_target_mode(target_mode) == "risk_diffuse"


# =============================================================================
# Dataset
# =============================================================================

class GraphDataset(Dataset):
    def __init__(
            self,
            Y_hist_raw: np.ndarray,
            Y_hist_target: np.ndarray,
            Y_target: np.ndarray,
            Y_rank: np.ndarray,
            Y_event: np.ndarray,
            X_hist_freq: np.ndarray,
            X_static_public: np.ndarray,
            X_structural: np.ndarray,
            calendar: np.ndarray,
            weather_by_node: np.ndarray,
            start_indices: List[int],
            hist_source_mode: str,
            history_len: int,
            horizon: int,
            use_priors: bool,
    ):
        super().__init__()
        self.Y_hist_raw = np.ascontiguousarray(Y_hist_raw.astype(np.float32, copy=False))
        self.Y_hist_target = np.ascontiguousarray(Y_hist_target.astype(np.float32, copy=False))
        self.Y_target = np.ascontiguousarray(Y_target.astype(np.float32, copy=False))
        self.Y_rank = np.ascontiguousarray(Y_rank.astype(np.float32, copy=False))
        self.Y_event = np.ascontiguousarray(Y_event.astype(np.float32, copy=False))
        self.X_hist_freq = np.ascontiguousarray(X_hist_freq.astype(np.float32, copy=False))
        self.X_static_public = np.ascontiguousarray(X_static_public.astype(np.float32, copy=False))
        self.X_structural = np.ascontiguousarray(X_structural.astype(np.float32, copy=False))
        self.calendar = np.ascontiguousarray(calendar.astype(np.float32, copy=False))
        self.weather_by_node = np.ascontiguousarray(weather_by_node.astype(np.float32, copy=False))
        self.start_indices = start_indices
        self.hist_source_mode = str(hist_source_mode)
        self.t = int(history_len)
        self.p = int(horizon)
        self.use_priors = bool(use_priors)
        self.num_nodes = int(self.Y_target.shape[0])

    def __len__(self) -> int:
        return len(self.start_indices)

    def _window_history(self, s: int) -> np.ndarray:
        raw_hist = self.Y_hist_raw[:, s:s + self.t, None]
        if self.hist_source_mode == "raw_risk":
            return raw_hist
        target_hist = self.Y_hist_target[:, s:s + self.t, None]
        if self.hist_source_mode == "target":
            return target_hist
        if self.hist_source_mode == "both":
            return np.concatenate([raw_hist, target_hist], axis=-1)
        raise ValueError(f"Unsupported hist_source_mode: {self.hist_source_mode}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.start_indices[idx]
        y_hist = self._window_history(s)
        y_fut = self.Y_target[:, s + self.t:s + self.t + self.p]
        y_rank_fut = self.Y_rank[:, s + self.t:s + self.t + self.p]
        y_event_fut = self.Y_event[:, s + self.t:s + self.t + self.p]

        x_static_t = np.broadcast_to(
            self.X_static_public[:, None, :],
            (self.num_nodes, self.t, self.X_static_public.shape[1]),
        )
        cal_hist = self.calendar[s:s + self.t, :]
        cal_t = np.broadcast_to(cal_hist[None, :, :], (self.num_nodes, self.t, cal_hist.shape[1]))
        w_hist = self.weather_by_node[:, s:s + self.t, :]
        x_hist_freq = self.X_hist_freq[:, s:s + self.t, None]
        if not self.use_priors:
            x_hist_freq = np.zeros_like(x_hist_freq, dtype=np.float32)

        parts = [y_hist]
        if self.use_priors:
            parts.append(x_hist_freq)
        if x_static_t.shape[-1] > 0:
            parts.append(x_static_t)
        if self.use_priors and self.X_structural.shape[1] > 0:
            x_structural_t = np.broadcast_to(
                self.X_structural[:, None, :],
                (self.num_nodes, self.t, self.X_structural.shape[1]),
            )
            parts.append(x_structural_t)
        if cal_t.shape[-1] > 0:
            parts.append(cal_t)
        if w_hist.shape[-1] > 0:
            parts.append(w_hist)
        x = np.concatenate(parts, axis=-1)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "y": torch.from_numpy(np.ascontiguousarray(y_fut)).float(),
            "y_rank": torch.from_numpy(np.ascontiguousarray(y_rank_fut)).float(),
            "y_event": torch.from_numpy(np.ascontiguousarray(y_event_fut)).float(),
            "y_event_rank": torch.from_numpy(np.ascontiguousarray(y_event_fut)).float(),
        }


# =============================================================================
# Bundle construction
# =============================================================================

def build_bundle(
        data_dir: str,
        start_date: str,
        end_date: str,
        history_len: int,
        horizon: int,
        batch_size: int,
        diffuse_alpha: float,
        diffuse_k: int,
        diffuse_weight: float,
        diffuse_norm: str,
        risk_diffuse_zero_thresh: float,
        device: torch.device,
        max_nodes: int = 0,
) -> dict:
    target_mode = "risk_diffuse"
    hist_source_mode = "raw_risk"
    use_priors = True
    use_hist_freq_prior = True
    use_structural_prior = True
    start_dt = parse_date_arg(start_date)
    end_dt = parse_date_arg(end_date)

    X_static, Y_risk, Y_crash, calendar, weather, borough_id_arr, _ = load_nyc_zinb_data(
        data_dir, start_dt, end_dt,
    )
    edge_index = np.load(os.path.join(data_dir, "line_graph", "edge_index.npy")).astype(np.int64)

    Y_risk_raw = Y_risk.copy()
    Y_hist_full = Y_risk_raw.copy()
    Y_crash_full = Y_crash.copy()

    num_nodes, total_steps = Y_hist_full.shape
    if total_steps < history_len + horizon:
        raise ValueError("Not enough time steps for the given history_len and horizon.")

    mask = np.ones(num_nodes, dtype=bool)
    if 0 < max_nodes < num_nodes:
        rng = np.random.default_rng(42)
        selected = rng.choice(num_nodes, size=max_nodes, replace=False)
        mask = np.zeros(num_nodes, dtype=bool)
        mask[selected] = True
        print(f"Sampling {max_nodes} nodes out of {num_nodes} for testing.")

    idx = np.where(mask)[0]

    X_static_public_sub = X_static[idx].copy().astype(np.float32)
    Y_hist_sub = Y_hist_full[idx].copy()
    Y_risk_sub = Y_risk[idx].copy()
    Y_crash_sub = Y_crash_full[idx].copy()
    borough_id_sub = borough_id_arr[idx].astype(np.int64, copy=True)

    edge_index_sub = apply_mask_to_edge_index(edge_index, mask)
    edge_index_undirected_sub = make_undirected_edge_index(edge_index_sub)

    structural_dim = 2
    if use_structural_prior:
        X_structural_sub = compute_structural_node_features(edge_index_undirected_sub, idx.size)
    else:
        X_structural_sub = np.zeros((idx.size, structural_dim), dtype=np.float32)

    weather_by_node_sub = weather[borough_id_sub].astype(np.float32)

    train_end = int(total_steps * (8 / 12))
    calendar, weather = standardize_time_features(calendar, weather, train_end)

    if use_hist_freq_prior:
        X_hist_freq_sub = build_exclusive_cumulative_mean(Y_crash_sub)
        X_hist_freq_sub, hist_mean, hist_std = standardize_node_time_feature(X_hist_freq_sub, train_end)
    else:
        X_hist_freq_sub = np.zeros_like(Y_crash_sub, dtype=np.float32)
        hist_mean, hist_std = 0.0, 1.0

    if uses_diffused_target(target_mode):
        Y_target_sub = diffuse_risk_time_space(
            Y_risk_sub,
            edge_index_sub,
            alpha=diffuse_alpha,
            k=diffuse_k,
            hop_weight=diffuse_weight,
            undirected=True,
            normalize=diffuse_norm,
            zero_thresh=risk_diffuse_zero_thresh,
        ).astype(np.float32)
    else:
        Y_target_sub = Y_risk_sub.copy()

    Y_hist_raw_sub = Y_hist_sub.copy()
    Y_hist_target_sub = Y_target_sub.copy()
    graph_hist_source_mode = hist_source_mode
    if hist_source_mode == "both" and not uses_diffused_target(target_mode):
        graph_hist_source_mode = "raw_risk"

    Y_rank_sub = Y_target_sub.copy()
    Y_event_sub = Y_risk_sub.copy()

    train_s, val_s, test_s = build_time_splits(total_steps, history_len, horizon)
    if len(train_s) == 0 or len(val_s) == 0 or len(test_s) == 0:
        raise ValueError("Time split resulted in empty subset.")

    shared_kwargs = dict(
        Y_hist_raw=Y_hist_raw_sub,
        Y_hist_target=Y_hist_target_sub,
        Y_target=Y_target_sub,
        Y_rank=Y_rank_sub,
        Y_event=Y_event_sub,
        X_hist_freq=X_hist_freq_sub,
        X_static_public=X_static_public_sub,
        X_structural=X_structural_sub,
        calendar=calendar,
        weather_by_node=weather_by_node_sub,
        hist_source_mode=graph_hist_source_mode,
        history_len=history_len,
        horizon=horizon,
    )

    train_ds = GraphDataset(start_indices=train_s, use_priors=use_priors, **shared_kwargs)
    val_ds = GraphDataset(start_indices=val_s, use_priors=use_priors, **shared_kwargs)
    test_ds = GraphDataset(start_indices=test_s, use_priors=use_priors, **shared_kwargs)

    A_q_sub, A_h_sub = build_diffusion_supports(edge_index_undirected_sub, idx.size, device)

    public_static_dim = int(X_static_public_sub.shape[1])
    structural_role_dim = int(X_structural_sub.shape[1])
    calendar_dim = int(calendar.shape[1])
    weather_dim = int(weather_by_node_sub.shape[2])
    hist_channels = 2 if graph_hist_source_mode == "both" else 1
    base_graph_feature_dim = hist_channels + public_static_dim + calendar_dim + weather_dim
    graph_input_dim = base_graph_feature_dim
    if use_priors:
        graph_input_dim += 1
        graph_input_dim += structural_role_dim

    train_label_idx = (
            np.asarray(train_s, dtype=np.int64)[:, None]
            + int(history_len)
            + np.arange(int(horizon), dtype=np.int64)[None, :]
    ).reshape(-1)
    train_label_block = np.take(Y_target_sub, train_label_idx, axis=1)
    train_event_block = np.take(Y_event_sub, train_label_idx, axis=1)
    train_event_pos_count = int(np.sum(train_event_block > 0))
    train_event_neg_count = int(train_event_block.size - train_event_pos_count)

    return {
        "num_nodes": int(idx.size),
        "graph_input_dim": int(graph_input_dim),
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "A_q": A_q_sub,
        "A_h": A_h_sub,
        "train_event_pos_count": train_event_pos_count,
        "train_event_neg_count": train_event_neg_count,
    }


# =============================================================================
# Model helper functions
# =============================================================================

def hurdle_gamma_mean(q_logits: torch.Tensor, mu: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    return q * mu.clamp_min(eps)


def hurdle_lognormal_mean(
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    exp_term = torch.clamp(mu_log + 0.5 * sigma_log * sigma_log, max=LOGNORMAL_EXP_CLAMP)
    cond_mean = torch.exp(exp_term)
    return q * cond_mean


@torch.no_grad()
def sample_hurdle_gamma(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        n_samples: int = 100,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    mu = mu.clamp_min(eps)
    alpha = alpha.clamp_min(eps)
    beta = alpha / mu
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
    shape = (int(n_samples),) + tuple(q.shape)
    bern = torch.bernoulli(q.expand(shape))
    gamma_samples = gamma_dist.sample((int(n_samples),))
    return bern * gamma_samples


@torch.no_grad()
def sample_hurdle_lognormal(
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        n_samples: int = 100,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    sigma_log = sigma_log.clamp_min(eps)
    shape = (int(n_samples),) + tuple(q.shape)
    bern = torch.bernoulli(q.expand(shape))
    eps_std = torch.randn(shape, device=q.device, dtype=q.dtype)
    lognormal_samples = torch.exp(
        torch.clamp(
            mu_log.unsqueeze(0) + sigma_log.unsqueeze(0) * eps_std,
            min=-LOGNORMAL_EXP_CLAMP,
            max=LOGNORMAL_EXP_CLAMP,
        )
    )
    return bern * lognormal_samples


def hurdle_gamma_interval_from_samples(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha_param: torch.Tensor,
        alpha: float,
        num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_hurdle_gamma(
        q_logits=q_logits, mu=mu, alpha=alpha_param, n_samples=num_samples,
    ).detach().cpu().numpy()
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def hurdle_lognormal_interval_from_samples(
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        alpha: float,
        num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_hurdle_lognormal(
        q_logits=q_logits, mu_log=mu_log, sigma_log=sigma_log, n_samples=num_samples,
    ).detach().cpu().numpy()
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


# =============================================================================
# Metrics
# =============================================================================

def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def picp(truth, L, U):
    y = _to_numpy(truth)
    L = _to_numpy(L)
    U = _to_numpy(U)
    return float(np.mean((y >= L) & (y <= U)))


def mpiw(L, U):
    L = _to_numpy(L)
    U = _to_numpy(U)
    return float(np.mean(U - L))


def interval_score(truth, L, U, alpha=0.1):
    y = _to_numpy(truth)
    L = _to_numpy(L)
    U = _to_numpy(U)
    width = U - L
    under = np.clip(L - y, a_min=0.0, a_max=None)
    over = np.clip(y - U, a_min=0.0, a_max=None)
    return float(np.mean(width + (2.0 / alpha) * under + (2.0 / alpha) * over))


def ndcg_at_k(truth, risk_score, top_frac=0.2):
    y = _to_numpy(truth).reshape(-1)
    score = _to_numpy(risk_score).reshape(-1)
    assert y.shape[0] == score.shape[0]
    N = score.shape[0]
    k = max(1, int(np.ceil(top_frac * N)))
    order = np.argsort(-score, kind="mergesort")[:k]
    rel = (y[order] > 0).astype(np.float64)
    total_pos = int(np.sum(y > 0))
    if total_pos == 0:
        return np.nan
    denom = min(total_pos, k)
    if denom == 0:
        return np.nan
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel * discounts))
    idcg = float(np.sum(discounts[:denom]))
    if idcg == 0:
        return np.nan
    return float(dcg / idcg)
