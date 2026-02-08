from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .features import bucket_key, extract_ts_features, zscore_rows
from .utils import as_1d_float_array, safe_str, DebugConfig


def simple_kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(max(1, min(k, n)))
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
            else:
                C[j] = X[rng.integers(0, n)]
    return labels


def cosine_to_centroid(batch_feats: np.ndarray) -> float:
    X = np.asarray(batch_feats, dtype=np.float32)
    if X.shape[0] <= 1:
        return 1.0
    c = X.mean(axis=0, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    sims = (Xn * cn).sum(axis=1)
    return float(np.mean(sims))


def build_item_representatives(test_entries: List[dict]) -> Dict[str, np.ndarray]:
    rep: Dict[str, np.ndarray] = {}
    rep_len: Dict[str, int] = {}
    for i, e in enumerate(test_entries):
        item_id = safe_str(e.get("item_id", f"series_{i}"))
        y = as_1d_float_array(e["target"])
        L = int(y.size)
        if (item_id not in rep) or (L > rep_len[item_id]):
            rep[item_id] = y
            rep_len[item_id] = L
    return rep


@dataclass(frozen=True)
class SemanticBatchingConfig:
    grouping: str = "features"  # features|bucket
    num_clusters: int = 50
    coherence_gate: bool = True
    coherence_threshold: float = 0.25
    kmeans_iters: int = 25
    seed: int = 0


@dataclass
class SemanticBatchPlan:
    ordered_batches: List[List[int]]
    item_to_group: Dict[str, str]
    item_feat_for_gate: Dict[str, np.ndarray]


class SemanticBatcher:
    def __init__(self, cfg: SemanticBatchingConfig, debug: DebugConfig):
        self.cfg = cfg
        self.debug = debug

    def plan(self, test_entries: List[dict], itemid_to_sem: Dict[str, Optional[str]], semantic_field: Optional[str], batch_size: int) -> SemanticBatchPlan:
        item_to_rep = build_item_representatives(test_entries)
        item_ids = sorted(item_to_rep.keys())

        # features for coherence gate and clustering
        item_feat_for_gate: Dict[str, np.ndarray] = {iid: extract_ts_features(item_to_rep[iid]) for iid in item_ids}

        item_to_group: Dict[str, str] = {}

        # semantic_field grouping (if present)
        if semantic_field is not None:
            for iid in item_ids:
                sem = itemid_to_sem.get(iid, None)
                if sem is not None and safe_str(sem) != "":
                    item_to_group[iid] = f"S:{safe_str(sem)}"

        remaining = [iid for iid in item_ids if iid not in item_to_group]

        if remaining:
            if self.cfg.grouping == "bucket":
                for iid in remaining:
                    item_to_group[iid] = f"K:{bucket_key(item_to_rep[iid])}"
            elif self.cfg.grouping == "features":
                feats = np.stack([item_feat_for_gate[iid] for iid in remaining], axis=0)
                feats_z, _, _ = zscore_rows(feats)

                # clamp k to <= n_items and <= unique rows
                n = feats_z.shape[0]
                unique_rows = np.unique(np.round(feats_z, 6), axis=0).shape[0]
                k = int(min(self.cfg.num_clusters, n, unique_rows))
                k = max(1, k)

                labels = simple_kmeans(feats_z, k=k, iters=self.cfg.kmeans_iters, seed=self.cfg.seed)
                for iid, lab in zip(remaining, labels):
                    item_to_group[iid] = f"F:{int(lab)}"
            else:
                raise ValueError("grouping must be 'features' or 'bucket'")

        ordered_batches = self._make_batches_from_groups(test_entries, item_to_group, batch_size)

        return SemanticBatchPlan(
            ordered_batches=ordered_batches,
            item_to_group=item_to_group,
            item_feat_for_gate=item_feat_for_gate,
        )

    @staticmethod
    def _make_batches_from_groups(test_entries: List[dict], item_to_group: Dict[str, str], batch_size: int) -> List[List[int]]:
        groups: Dict[str, List[int]] = {}
        for idx, e in enumerate(test_entries):
            item_id = safe_str(e.get("item_id", f"series_{idx}"))
            g = item_to_group.get(item_id, "G:unknown")
            groups.setdefault(g, []).append(idx)

        ordered_batches: List[List[int]] = []
        for _, idxs in groups.items():
            idxs.sort(key=lambda k: len(as_1d_float_array(test_entries[k]["target"])))
            for j in range(0, len(idxs), batch_size):
                ordered_batches.append(idxs[j: j + batch_size])

        return ordered_batches