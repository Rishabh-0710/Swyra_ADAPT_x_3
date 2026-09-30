"""
Streaming feature encoder with a frozen schema and bounded state.

Design rules (each one is covered by a test):

1. **Frozen schema.** The output columns are fixed when batch 0 is seen.
   Every numeric feature always owns a missing-indicator column, whether or
   not a given batch contains missing values, so dimensionality never changes.
2. **Stable feature semantics.** Numeric values are passed through unscaled
   (tree learners are invariant to monotone rescaling). Scaling statistics
   that drift over time would silently change the meaning of every tree that
   the persistent model state already holds, so they are not used.
3. **Bounded categorical state.** Each categorical feature keeps a frozen
   vocabulary of at most ``max_categories`` values from batch 0. Values not
   in the vocabulary are hashed (CRC32, deterministic across processes) into
   ``n_hash_buckets`` overflow codes. The number of codes is therefore fixed
   and independent of how many distinct categories the stream produces.
4. **No target leakage.** Optional target encoding is a two-phase process:
   ``transform`` only reads statistics accumulated from *earlier* batches and
   ``observe(X, y)`` is called only after the model update for that batch has
   finished. Batch 0 uses out-of-fold encoding so no row sees its own label.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.task import TaskSpec

ArrayLike = Union[pd.DataFrame, np.ndarray]


def _as_frame(X: ArrayLike, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = columns if columns is not None and len(columns) == arr.shape[1] else [f"feat_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _stable_hash(value: str, n_buckets: int) -> int:
    return zlib.crc32(value.encode("utf-8")) % n_buckets


class StreamingFeatureEncoder:
    """Encodes raw tabular batches into a fixed numeric matrix."""

    def __init__(
        self,
        task_spec: TaskSpec,
        max_categories: int = 64,
        n_hash_buckets: int = 16,
        use_target_encoding: bool = False,
        te_smoothing: float = 20.0,
        te_decay: float = 0.9,
        monitor_decay: float = 0.8,
        random_seed: int = 0,
    ) -> None:
        if max_categories + 1 + n_hash_buckets > 250:
            raise ValueError("Categorical code space must fit in 250 codes (histogram GBDT limit).")
        self.task_spec = task_spec
        self.max_categories = int(max_categories)
        self.n_hash_buckets = int(n_hash_buckets)
        self.use_target_encoding = bool(use_target_encoding)
        self.te_smoothing = float(te_smoothing)
        self.te_decay = float(te_decay)
        self.monitor_decay = float(monitor_decay)
        self.random_seed = random_seed

        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.vocab: Dict[str, Dict[str, int]] = {}
        self.feature_names_out_: List[str] = []
        self.categorical_mask_: Optional[np.ndarray] = None
        self.numeric_columns_: Optional[np.ndarray] = None
        self.distance_scale_: Optional[np.ndarray] = None
        self.n_codes_: Dict[str, int] = {}

        # bounded streaming statistics (arrays sized by the frozen schema)
        self.te_sum_: Dict[str, np.ndarray] = {}
        self.te_cnt_: Dict[str, np.ndarray] = {}
        self.global_sum_: float = 0.0
        self.global_cnt_: float = 0.0
        self.missing_rate_: Optional[np.ndarray] = None
        self.novel_category_rate_: Dict[str, float] = {}
        self.n_observed_batches_: int = 0
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ schema
    def fit_schema(self, X: ArrayLike) -> "StreamingFeatureEncoder":
        """Freeze the output schema using batch 0 inputs only (labels are not used)."""
        df = _as_frame(X)
        spec = self.task_spec
        if not spec.numerical_features and not spec.categorical_features:
            spec.numerical_features = [c for c in df.columns if c != spec.subgroup_col and pd.api.types.is_numeric_dtype(df[c])]
            spec.categorical_features = [c for c in df.columns if c != spec.subgroup_col and c not in spec.numerical_features]
        self.numerical_features = list(spec.numerical_features)
        self.categorical_features = list(spec.categorical_features)

        for col in self.categorical_features:
            series = df[col] if col in df.columns else pd.Series([None] * len(df))
            counts = series.dropna().astype(str).value_counts()
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_categories]
            self.vocab[col] = {cat: i for i, (cat, _) in enumerate(ranked)}
            self.n_codes_[col] = len(self.vocab[col]) + 1 + self.n_hash_buckets
            if self.use_target_encoding:
                self.te_sum_[col] = np.zeros(self.n_codes_[col])
                self.te_cnt_[col] = np.zeros(self.n_codes_[col])

        names = list(self.numerical_features)
        names += [f"{c}__code" for c in self.categorical_features]
        names += [f"{c}__isna" for c in self.numerical_features]
        if self.use_target_encoding:
            names += [f"{c}__target_enc" for c in self.categorical_features]
        self.feature_names_out_ = names

        n_num, n_cat = len(self.numerical_features), len(self.categorical_features)
        mask = np.zeros(len(names), dtype=bool)
        mask[n_num: n_num + n_cat] = True
        self.categorical_mask_ = mask
        self.numeric_columns_ = np.arange(n_num)

        num = self._numeric_block(df)
        q75, q25 = np.nanpercentile(num, [75, 25], axis=0) if n_num else (np.array([]), np.array([]))
        scale = (q75 - q25) / 1.349 if n_num else np.array([])
        std = np.nanstd(num, axis=0) if n_num else np.array([])
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.where(std > 1e-9, std, 1.0))
        self.distance_scale_ = np.nan_to_num(scale, nan=1.0)
        self.missing_rate_ = np.zeros(n_num + n_cat)
        self.is_fitted = True
        return self

    # --------------------------------------------------------------- encoding
    def _numeric_block(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.numerical_features:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=True)
            else:
                v = np.full(len(df), np.nan)
            v[~np.isfinite(v)] = np.nan
            cols.append(v)
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def _category_codes(self, df: pd.DataFrame, col: str) -> np.ndarray:
        vocab = self.vocab[col]
        n_vocab = len(vocab)
        series = df[col] if col in df.columns else pd.Series([None] * len(df))
        missing = series.isna().to_numpy()
        as_str = series.astype(str)
        codes = np.array(as_str.map(vocab), dtype=float)
        unseen = np.isnan(codes) & ~missing
        if unseen.any():
            uniq = pd.unique(as_str[unseen])
            lut = {u: n_vocab + 1 + _stable_hash(u, self.n_hash_buckets) for u in uniq}
            codes[unseen] = as_str[unseen].map(lut).to_numpy(dtype=float)
        codes[missing] = n_vocab
        return codes

    def _target_encode(self, col: str, codes: np.ndarray, te_sum: np.ndarray, te_cnt: np.ndarray, g_sum: float, g_cnt: float) -> np.ndarray:
        prior = g_sum / g_cnt if g_cnt > 0 else 0.0
        c = codes.astype(int)
        return (te_sum[c] + self.te_smoothing * prior) / (te_cnt[c] + self.te_smoothing)

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode X using only state accumulated from previously observed batches."""
        if not self.is_fitted:
            raise RuntimeError("Encoder schema is not fitted.")
        df = _as_frame(X, self.numerical_features + self.categorical_features)
        num = self._numeric_block(df)
        codes = [self._category_codes(df, c) for c in self.categorical_features]
        blocks = [num]
        if codes:
            blocks.append(np.column_stack(codes))
        blocks.append(np.isnan(num).astype(float))
        if self.use_target_encoding and codes:
            te = [
                self._target_encode(c, codes[i], self.te_sum_[c], self.te_cnt_[c], self.global_sum_, self.global_cnt_)
                for i, c in enumerate(self.categorical_features)
            ]
            blocks.append(np.column_stack(te))
        out = np.hstack(blocks).astype(float)
        assert out.shape[1] == len(self.feature_names_out_), "schema violation"
        return out

    def initial_transform(self, X: ArrayLike, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
        """Batch-0 encoding. With target encoding, every row is encoded out-of-fold."""
        Z = self.transform(X)
        if not (self.use_target_encoding and self.categorical_features):
            return Z
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_seed)
        folds = rng.permutation(len(y)) % n_folds
        n_num, n_cat = len(self.numerical_features), len(self.categorical_features)
        te_start = 2 * n_num + n_cat
        for j, col in enumerate(self.categorical_features):
            codes = Z[:, n_num + j].astype(int)
            for f in range(n_folds):
                train, test = folds != f, folds == f
                s = np.bincount(codes[train], weights=y[train], minlength=self.n_codes_[col])
                n = np.bincount(codes[train], minlength=self.n_codes_[col]).astype(float)
                Z[test, te_start + j] = self._target_encode(col, codes[test], s, n, y[train].sum(), float(train.sum()))
        return Z

    # ------------------------------------------------------ post-update state
    def observe(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> None:
        """Absorb a batch *after* the model has been updated with it."""
        df = _as_frame(X, self.numerical_features + self.categorical_features)
        d = self.monitor_decay
        num = self._numeric_block(df)
        rates = [np.isnan(num).mean(axis=0)] if num.shape[1] else []
        for col in self.categorical_features:
            codes = self._category_codes(df, col)
            n_vocab = len(self.vocab[col])
            rates.append(np.array([np.mean(codes == n_vocab)]))
            novel = float(np.mean(codes > n_vocab)) if len(codes) else 0.0
            self.novel_category_rate_[col] = d * self.novel_category_rate_.get(col, novel) + (1 - d) * novel
            if self.use_target_encoding and y is not None:
                yy = np.asarray(y, dtype=float)
                c = codes.astype(int)
                self.te_sum_[col] = self.te_decay * self.te_sum_[col] + np.bincount(c, weights=yy, minlength=self.n_codes_[col])
                self.te_cnt_[col] = self.te_decay * self.te_cnt_[col] + np.bincount(c, minlength=self.n_codes_[col])
        if y is not None and self.use_target_encoding:
            yy = np.asarray(y, dtype=float)
            self.global_sum_ = self.te_decay * self.global_sum_ + yy.sum()
            self.global_cnt_ = self.te_decay * self.global_cnt_ + len(yy)
        if rates:
            new_rate = np.concatenate(rates)
            self.missing_rate_ = new_rate if self.n_observed_batches_ == 0 else d * self.missing_rate_ + (1 - d) * new_rate
        self.n_observed_batches_ += 1

    # ------------------------------------------------------------- accounting
    def state_nbytes(self) -> int:
        from src.memory_accounting import deep_nbytes

        return deep_nbytes(self)
