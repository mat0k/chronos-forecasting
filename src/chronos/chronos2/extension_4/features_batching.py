#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core_utils import (
    as_1d_float_array,
    batch_sanity_stats,
    is_fragmented,
    pack_batches_global,
    safe_str,
)


# -----------------------------
# Feature extraction
# -----------------------------
class FeatureExtractor:
    def seasonality_strength(self, x: np.ndarray, max_lag: int = 24) -> float:
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        if n < 8:
            return 0.0
        x = x - x.mean()
        denom = float((x * x).sum())
        if denom <= 1e-12:
            return 0.0

        max_lag = int(min(max_lag, n // 2))
        best = 0.0
        for lag in range(2, max_lag + 1):
            a = x[:-lag]
            b = x[lag:]
            num = float((a * b).sum())
            ac = abs(num / denom)
            if ac > best:
                best = ac
        return float(max(0.0, min(1.0, best)))

    def _safe_skew(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        if x.size < 3:
            return 0.0
        m = float(np.mean(x))
        s = float(np.std(x) + 1e-8)
        return float(np.mean(((x - m) / s) ** 3))

    def _safe_kurtosis(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        if x.size < 4:
            return 0.0
        m = float(np.mean(x))
        s = float(np.std(x) + 1e-8)
        return float(np.mean(((x - m) / s) ** 4) - 3.0)

    def _linear_slope(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        if n < 4:
            return 0.0
        t = np.arange(n, dtype=np.float32)
        t = (t - t.mean()) / (t.std() + 1e-8)
        y = x - x.mean()
        denom = float((t * t).sum()) + 1e-8
        return float((t * y).sum() / denom)

    def _acf_at_lag(self, x: np.ndarray, lag: int) -> float:
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        if n <= lag + 2:
            return 0.0
        x = x - x.mean()
        denom = float((x * x).sum()) + 1e-8
        num = float((x[:-lag] * x[lag:]).sum())
        return float(num / denom)

    def _dominant_period_fft(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        if n < 16:
            return 0.0
        y = x - x.mean()
        spec = np.fft.rfft(y)
        mag = np.abs(spec)
        if mag.size <= 2:
            return 0.0
        mag[0] = 0.0
        k = int(np.argmax(mag))
        if k <= 0:
            return 0.0
        period = float(n / k)
        return float(max(1.0, min(period, float(n))))

    def extract_ts_features(self, y: np.ndarray, max_lag: int = 24) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((9 + max_lag,), dtype=np.float32)

        tail = y[-min(y.size, 256):]
        m = float(np.mean(tail))
        s = float(np.std(tail) + 1e-8)
        log_s = float(np.log10(s + 1e-6))

        slope = self._linear_slope(tail)
        skew = self._safe_skew(tail)
        kurt = self._safe_kurtosis(tail)
        domp = self._dominant_period_fft(tail)
        seas = self.seasonality_strength(tail, max_lag=max_lag)

        acfs = [self._acf_at_lag(tail, lag) for lag in range(1, max_lag + 1)]
        acf_max = float(max(np.abs(acfs))) if acfs else 0.0

        feat = np.array(
            [m, s, log_s, slope, skew, kurt, domp, seas, acf_max] + acfs,
            dtype=np.float32,
        )
        return np.clip(feat, -50.0, 50.0)

    @staticmethod
    def zscore_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mu) / sd, mu, sd

    @staticmethod
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

    @staticmethod
    def cosine_to_centroid(batch_feats: np.ndarray) -> float:
        X = batch_feats.astype(np.float32)
        if X.shape[0] <= 1:
            return 1.0
        c = X.mean(axis=0, keepdims=True)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
        sims = (Xn * cn).sum(axis=1)
        return float(np.mean(sims))


# -----------------------------
# Bucketing utilities
# -----------------------------
class Bucketizer:
    def __init__(self, feat: FeatureExtractor):
        self.feat = feat

    def bucket_key(
        self,
        target: np.ndarray,
        length_bins: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        scale_bins: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        seas_bins: Tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.70),
    ) -> str:
        y = np.asarray(target, dtype=np.float32).reshape(-1)
        L = int(y.size)
        tail = y[-min(L, 64):] if L > 0 else y
        std = float(np.std(tail, ddof=0)) if L > 0 else 0.0
        log_std = float(np.log10(std + 1e-6))
        seas = self.feat.seasonality_strength(y[-min(L, 256):], max_lag=24)

        lb = 0
        for t in length_bins:
            if L <= t:
                break
            lb += 1

        sb = 0
        for t in scale_bins:
            if log_std <= t:
                break
            sb += 1

        zb = 0
        for t in seas_bins:
            if seas <= t:
                break
            zb += 1

        return f"B_len{lb}_s{sb}_z{zb}"

    def enhanced_bucket_key(
        self,
        target: np.ndarray,
        length_bins: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        scale_bins: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        seas_bins: Tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.70),
        slope_bins: Tuple[float, ...] = (-0.5, -0.2, -0.05, 0.05, 0.2, 0.5),
        period_bins: Tuple[float, ...] = (3, 6, 12, 24, 48, 96, 168),
    ) -> str:
        y = np.asarray(target, dtype=np.float32).reshape(-1)
        L = int(y.size)
        tail = y[-min(L, 256):] if L > 0 else y
        std = float(np.std(tail[-min(L, 64):], ddof=0)) if L > 0 else 0.0
        log_std = float(np.log10(std + 1e-6))
        seas = self.feat.seasonality_strength(tail, max_lag=24)
        slope = self.feat._linear_slope(tail)
        domp = self.feat._dominant_period_fft(tail)

        lb = 0
        for t in length_bins:
            if L <= t:
                break
            lb += 1

        sb = 0
        for t in scale_bins:
            if log_std <= t:
                break
            sb += 1

        zb = 0
        for t in seas_bins:
            if seas <= t:
                break
            zb += 1

        tb = 0
        for t in slope_bins:
            if slope <= t:
                break
            tb += 1

        pb = 0
        for t in period_bins:
            if domp <= t:
                break
            pb += 1

        return f"E_len{lb}_s{sb}_z{zb}_t{tb}_p{pb}"


# -----------------------------
# Batching planner (neighbors + fallback packing)
# -----------------------------
@dataclass
class BatchingConfig:
    batch_size: int = 32
    neighbor_top_k: int = 64
    neighbor_threshold: float = 0.20
    max_group_for_bruteforce: int = 5000
    min_batch_fill: float = 0.5
    max_batch_overhead: float = 3.0
    seed: int = 0


class SemanticBatchPlanner:
    def __init__(self, feat: FeatureExtractor, bucket: Bucketizer):
        self.feat = feat
        self.bucket = bucket

    @staticmethod
    def _normalize_rows_l2(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / nrm

    @staticmethod
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

    def make_batches_random_within_groups(
        self,
        test_entries: List[dict],
        item_to_group: Dict[str, str],
        batch_size: int,
        seed: int,
    ) -> List[List[int]]:
        rng = np.random.default_rng(seed)
        group_to_indices: Dict[str, List[int]] = {}
        for idx, e in enumerate(test_entries):
            iid = safe_str(e.get("item_id", f"series_{idx}"))
            g = item_to_group.get(iid, "G:unknown")
            group_to_indices.setdefault(g, []).append(idx)

        prelim: List[List[int]] = []
        for g in sorted(group_to_indices.keys()):
            idxs = group_to_indices[g]
            rng.shuffle(idxs)
            for j in range(0, len(idxs), batch_size):
                prelim.append(idxs[j: j + batch_size])

        return pack_batches_global(prelim, batch_size=batch_size)

    def _topk_neighbors_batching_within_group(
        self,
        group_item_ids: List[str],
        feats_by_item: Dict[str, np.ndarray],
        batch_size: int,
        top_k: int,
        sim_threshold: float,
        max_group_for_bruteforce: int,
        seed: int,
        min_batch_size: int,
    ) -> List[List[str]]:
        if len(group_item_ids) <= 1:
            return [group_item_ids]

        if len(group_item_ids) > max_group_for_bruteforce:
            return []

        rng = np.random.default_rng(seed)
        ids = list(group_item_ids)
        rng.shuffle(ids)

        X = np.stack([feats_by_item[iid] for iid in ids], axis=0)
        Xn = self._normalize_rows_l2(X)

        assigned = np.zeros(len(ids), dtype=bool)
        batches: List[List[str]] = []

        for i in range(len(ids)):
            if assigned[i]:
                continue

            seed_vec = Xn[i]
            sims = Xn @ seed_vec
            sims[i] = -1.0

            avail = np.where((~assigned) & (np.arange(len(ids)) != i))[0]
            if avail.size == 0:
                assigned[i] = True
                batches.append([ids[i]])
                continue

            order = avail[np.argsort(-sims[avail])]

            good = order[sims[order] >= sim_threshold]
            chosen = good[: int(min(top_k, good.size))].tolist() if good.size > 0 else []

            batch_idx = [i] + chosen

            # weak-fill to reach at least min_batch_size
            if len(batch_idx) < min_batch_size:
                for j in order:
                    if j in batch_idx:
                        continue
                    batch_idx.append(int(j))
                    if len(batch_idx) >= min_batch_size:
                        break

            # fill up to batch_size
            if len(batch_idx) < batch_size:
                for j in order:
                    if j in batch_idx:
                        continue
                    batch_idx.append(int(j))
                    if len(batch_idx) >= batch_size:
                        break

            for j in batch_idx:
                assigned[j] = True

            batches.append([ids[j] for j in batch_idx])

        return batches

    def make_batches_neighbor_batched(
        self,
        test_entries: List[dict],
        item_to_group: Dict[str, str],
        item_feats_z: Dict[str, np.ndarray],
        item_to_len: Dict[str, int],
        cfg: BatchingConfig,
    ) -> List[List[int]]:
        group_to_indices: Dict[str, List[int]] = {}
        for idx, e in enumerate(test_entries):
            iid = safe_str(e.get("item_id", f"series_{idx}"))
            g = item_to_group.get(iid, "G:unknown")
            group_to_indices.setdefault(g, []).append(idx)

        ordered_batches: List[List[int]] = []
        rng = np.random.default_rng(cfg.seed)
        min_batch_size = int(max(2, round(cfg.min_batch_fill * cfg.batch_size)))

        for g in sorted(group_to_indices.keys()):
            idxs = group_to_indices[g]
            iids = [safe_str(test_entries[i].get("item_id", f"series_{i}")) for i in idxs]

            if len(idxs) > cfg.max_group_for_bruteforce:
                idxs.sort(key=lambda k: item_to_len.get(safe_str(test_entries[k].get("item_id", "")), 0))
                for j in range(0, len(idxs), cfg.batch_size):
                    ordered_batches.append(idxs[j: j + cfg.batch_size])
                continue

            neigh_batches = self._topk_neighbors_batching_within_group(
                group_item_ids=iids,
                feats_by_item=item_feats_z,
                batch_size=cfg.batch_size,
                top_k=cfg.neighbor_top_k,
                sim_threshold=cfg.neighbor_threshold,
                max_group_for_bruteforce=cfg.max_group_for_bruteforce,
                seed=int(rng.integers(0, 2**31 - 1)),
                min_batch_size=min_batch_size,
            )

            if not neigh_batches:
                rng.shuffle(idxs)
                for j in range(0, len(idxs), cfg.batch_size):
                    ordered_batches.append(idxs[j: j + cfg.batch_size])
                continue

            iid_to_indices: Dict[str, List[int]] = {}
            for i in idxs:
                iid = safe_str(test_entries[i].get("item_id", f"series_{i}"))
                iid_to_indices.setdefault(iid, []).append(i)

            for b in neigh_batches:
                out: List[int] = []
                for iid in b:
                    if iid_to_indices.get(iid):
                        out.append(iid_to_indices[iid].pop())
                if out:
                    ordered_batches.append(out)

            leftovers = [i for v in iid_to_indices.values() for i in v]
            if leftovers:
                rng.shuffle(leftovers)
                for j in range(0, len(leftovers), cfg.batch_size):
                    ordered_batches.append(leftovers[j: j + cfg.batch_size])

        return ordered_batches

    def build_grouping(
        self,
        test_entries: List[dict],
        itemid_to_sem: Dict[str, Optional[str]],
        semantic_field: Optional[str],
        grouping: str,
        num_clusters: int,
        kmeans_iters: int,
        seed: int,
    ) -> Tuple[Dict[str, str], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
        """
        Returns:
          item_to_group, item_to_rep, item_feats_z, item_to_len
        """
        item_to_rep = self.build_item_representatives(test_entries)
        item_ids = sorted(item_to_rep.keys())
        item_to_len = {iid: int(item_to_rep[iid].size) for iid in item_ids}

        item_to_group: Dict[str, str] = {}

        # semantic_field grouping priority
        if semantic_field is not None:
            for iid in item_ids:
                sem = itemid_to_sem.get(iid, None)
                if sem is not None and safe_str(sem) != "":
                    item_to_group[iid] = f"S:{safe_str(sem)}"

        remaining = [iid for iid in item_ids if iid not in item_to_group]
        mode = grouping.strip().lower()

        if remaining:
            if mode in {"neighbors", "enhanced_bucket", "enhanced-bucket"}:
                for iid in remaining:
                    item_to_group[iid] = f"E:{self.bucket.enhanced_bucket_key(item_to_rep[iid])}"
            elif mode in {"bucket"}:
                for iid in remaining:
                    item_to_group[iid] = f"K:{self.bucket.bucket_key(item_to_rep[iid])}"
            elif mode in {"features", "features_kmeans", "kmeans"}:
                feats = np.stack([self.feat.extract_ts_features(item_to_rep[iid]) for iid in remaining], axis=0)
                feats_z, _, _ = self.feat.zscore_rows(feats)
                k = int(min(num_clusters, feats_z.shape[0]))
                labels = self.feat.simple_kmeans(feats_z, k=k, iters=kmeans_iters, seed=seed)
                for iid, lab in zip(remaining, labels):
                    item_to_group[iid] = f"F:{int(lab)}"
            else:
                raise ValueError("grouping must be one of: neighbors | enhanced_bucket | bucket | features_kmeans")

        # z-scored features for all unique items (neighbor + coherence gate)
        item_feats = {iid: self.feat.extract_ts_features(item_to_rep[iid]) for iid in item_ids}
        feats_all = np.stack([item_feats[iid] for iid in item_ids], axis=0)
        feats_all_z, _, _ = self.feat.zscore_rows(feats_all)
        item_feats_z: Dict[str, np.ndarray] = {iid: feats_all_z[i] for i, iid in enumerate(item_ids)}

        return item_to_group, item_to_rep, item_feats_z, item_to_len

    def plan_batches(
        self,
        test_entries: List[dict],
        item_to_group: Dict[str, str],
        item_feats_z: Dict[str, np.ndarray],
        item_to_len: Dict[str, int],
        grouping: str,
        cfg: BatchingConfig,
    ) -> Tuple[List[List[int]], str]:
        """
        Returns (ordered_batches, batching_kind)
        """
        mode = grouping.strip().lower()
        N = len(test_entries)

        if mode in {"neighbors", "enhanced_bucket", "enhanced-bucket"}:
            ordered_batches = self.make_batches_neighbor_batched(
                test_entries=test_entries,
                item_to_group=item_to_group,
                item_feats_z=item_feats_z,
                item_to_len=item_to_len,
                cfg=cfg,
            )
            batching_kind = "topk-neighbors+weakfill"
        else:
            groups: Dict[str, List[int]] = {}
            for idx, e in enumerate(test_entries):
                iid = safe_str(e.get("item_id", f"series_{idx}"))
                g = item_to_group.get(iid, "G:unknown")
                groups.setdefault(g, []).append(idx)

            ordered_batches = []
            for _, idxs in groups.items():
                idxs.sort(key=lambda k: len(as_1d_float_array(test_entries[k]["target"])))
                for j in range(0, len(idxs), cfg.batch_size):
                    ordered_batches.append(idxs[j: j + cfg.batch_size])
            batching_kind = "length-sorted"

        if is_fragmented(
            ordered_batches,
            N=N,
            batch_size=cfg.batch_size,
            min_fill=cfg.min_batch_fill,
            max_overhead=cfg.max_batch_overhead,
        ):
            st = batch_sanity_stats(ordered_batches, N, cfg.batch_size)
            ordered_batches = self.make_batches_random_within_groups(
                test_entries=test_entries,
                item_to_group=item_to_group,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
            )
            batching_kind = (
                f"fallback(random-within-groups+packed; prev avg_bs={st['avg_bs']:.2f}, "
                f"actual={int(st['actual'])}, expected~={int(st['expected'])})"
            )

        return ordered_batches, batching_kind