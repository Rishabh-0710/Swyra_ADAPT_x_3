"""
Distribution-shift detection with a bounded reference.

The detector keeps a bounded reference instead of historical data:

* ``ref_X``: at most ``reference_capacity`` encoded rows (a random
  representative sample, refreshed after each update at the rate the
  adaptation controller chooses);
* ``ref_loss`` / ``ref_resid``: at most ``reference_capacity`` held-out
  per-row losses and residuals of the *current* model on the distribution it
  was last adapted to;
* per-feature quantile bin edges and bin probabilities for PSI;
* a two-float Page-Hinkley statistic over batches for gradual drift.

Its size depends only on ``reference_capacity`` and the number of features,
never on stream length or the size of the initial dataset.

Tests (every p-value is controlled for multiple comparisons):

* covariate shift, per feature: two-sample Kolmogorov-Smirnov (numeric) or
  chi-square (categorical codes, missingness). Benjamini-Hochberg FDR at
  ``alpha`` (default 1%) across features, plus a practical-significance gate (PSI >= 0.1,
  the usual "moderate shift" threshold);
* correlation (joint) shift: Fisher-z test on every pair of Spearman
  correlations, Bonferroni-corrected, gated on |delta rho| >= 0.2. This catches
  rotations that leave every marginal distribution unchanged;
* concept shift P(y|x): the current model's **pre-update** per-row loss on
  the incoming batch, restricted to rows inside the reference support, is
  compared with the reference held-out losses (one-sided Mann-Whitney U at
  ``alpha_model``, default 5%; it is a single test per batch), gated on a loss ratio >= 1 + ``loss_min_effect``. Restricting to in-support
  rows separates a changed mapping from extrapolation error caused by covariate
  shift;
* prior / label shift: a residual-bias test (Welch t-test for regression,
  calibration-in-the-large z-test for classification), called label shift when
  the bias explains >= 50% of the squared error (regression) or the logit of
  the base rate is off by >= 0.5 (classification).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def per_row_loss(task_type: str, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Squared error (regression) or log-loss (classification; pred = P(y=1))."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if task_type == "regression":
        return (y - pred) ** 2
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _psi(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _smoothed(counts: np.ndarray, eps: float = 0.5) -> np.ndarray:
    c = counts.astype(float) + eps
    return c / c.sum()


@dataclass
class FeatureShift:
    feature: str
    kind: str                  # numeric | categorical | missingness
    test: str
    statistic: float
    p_value: float
    q_value: float = 1.0
    psi: float = 0.0
    wasserstein: float = 0.0   # normalised by reference std (numeric only)
    drifted: bool = False


@dataclass
class ShiftReport:
    batch_index: int
    n_samples: int
    overall_drift_score: float = 0.0
    severity: str = "none"                 # none | low | medium | high
    drift_type: str = "none"               # none | covariate | concept | prior | compound | gradual
    covariate_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    covariate_drift: bool = False
    correlation_drift: bool = False
    concept_drift: bool = False
    prior_shift: bool = False
    gradual_drift: bool = False
    affected_features: List[str] = field(default_factory=list)
    feature_shifts: List[FeatureShift] = field(default_factory=list)
    correlation_max_delta: float = 0.0
    correlation_p_value: float = 1.0
    loss_ratio_in_support: float = 1.0
    loss_ratio_all: float = 1.0
    concept_p_value: float = 1.0
    in_support_fraction: float = 1.0
    bias_share: float = 0.0
    bias_p_value: float = 1.0
    cusum_statistic: float = 0.0
    confidence: float = 1.0

    def summary(self) -> str:
        parts = [f"batch {self.batch_index}: {self.severity.upper()} {self.drift_type} drift "
                 f"(score={self.overall_drift_score:.2f}, conf={self.confidence:.2f})"]
        if self.affected_features:
            parts.append("features: " + ", ".join(self.affected_features[:6]))
        if self.correlation_drift:
            parts.append(f"correlation shift |d_rho|={self.correlation_max_delta:.2f}")
        parts.append(f"pre-update loss ratio in-support={self.loss_ratio_in_support:.2f} "
                     f"(p={self.concept_p_value:.1e}, support={self.in_support_fraction:.0%})")
        if self.prior_shift:
            parts.append(f"label/prior shift (bias share {self.bias_share:.0%})")
        return "; ".join(parts)

    def to_dict(self, include_features: bool = False) -> Dict:
        d = asdict(self)
        if not include_features:
            d.pop("feature_shifts")
        return d


class DistributionShiftDetector:
    def __init__(
        self,
        task_type: str,
        feature_names: List[str],
        n_numeric: int,
        n_categorical: int,
        reference_capacity: int = 512,
        alpha: float = 0.01,
        alpha_model: float = 0.05,
        psi_min_effect: float = 0.10,
        corr_min_effect: float = 0.20,
        loss_min_effect: float = 0.15,
        min_support_rows: int = 30,
        n_bins: int = 10,
        random_seed: int = 0,
    ) -> None:
        self.task_type = task_type
        self.n_num = n_numeric
        self.n_cat = n_categorical
        # monitored columns: numeric | categorical codes | numeric missing indicators
        self.n_cols = 2 * n_numeric + n_categorical
        self.names = list(feature_names[: self.n_cols])
        self.capacity = int(reference_capacity)
        self.alpha = alpha                # FDR level across the many per-feature tests
        self.alpha_model = alpha_model    # level of the single concept and label-bias tests
        self.psi_min = psi_min_effect
        self.corr_min = corr_min_effect
        self.loss_min = loss_min_effect
        self.min_support = min_support_rows
        self.n_bins = n_bins
        self.rng = np.random.default_rng(random_seed)

        self.ref_X: Optional[np.ndarray] = None
        self.ref_loss: Optional[np.ndarray] = None
        self.ref_resid: Optional[np.ndarray] = None
        self.bin_edges: List[np.ndarray] = []
        self.ref_bin_p: List[np.ndarray] = []
        self.ref_std: Optional[np.ndarray] = None
        self.ref_corr: Optional[np.ndarray] = None
        self.corr_cols: Optional[np.ndarray] = None
        self.cusum: float = 0.0
        self.n_batches: int = 0

    # ------------------------------------------------------------ reference
    def _subsample(self, X: np.ndarray, k: int) -> np.ndarray:
        if len(X) <= k:
            return X.copy()
        return X[np.sort(self.rng.choice(len(X), size=k, replace=False))].copy()

    def _rebuild_summaries(self) -> None:
        X = self.ref_X
        self.bin_edges, self.ref_bin_p = [], []
        stds = np.ones(self.n_num)
        for j in range(self.n_num):
            v = X[:, j][~np.isnan(X[:, j])]
            if len(v) < 5:
                self.bin_edges.append(np.array([]))
                self.ref_bin_p.append(np.array([1.0]))
                continue
            edges = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.bin_edges.append(edges)
            self.ref_bin_p.append(_smoothed(np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)))
            s = np.std(v)
            stds[j] = s if s > 1e-12 else 1.0
        self.ref_std = stds
        num = X[:, : self.n_num]
        var_ok = np.array([np.nanstd(num[:, j]) > 1e-12 for j in range(self.n_num)], dtype=bool)
        self.corr_cols = np.where(var_ok)[0][:100]
        self.ref_corr = self._spearman(num[:, self.corr_cols]) if len(self.corr_cols) >= 2 else None

    @staticmethod
    def _spearman(M: np.ndarray) -> np.ndarray:
        M = M.copy()
        med = np.nanmedian(M, axis=0)
        idx = np.where(np.isnan(M))
        M[idx] = np.take(med, idx[1])
        R = np.apply_along_axis(stats.rankdata, 0, M)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(R, rowvar=False)
        return np.nan_to_num(C)

    def fit_reference(self, X: np.ndarray, heldout_loss: np.ndarray, heldout_resid: np.ndarray) -> None:
        self.ref_X = self._subsample(np.asarray(X)[:, : self.n_cols], self.capacity)
        self.ref_loss = self._subsample(np.asarray(heldout_loss).reshape(-1, 1), self.capacity).ravel()
        self.ref_resid = self._subsample(np.asarray(heldout_resid).reshape(-1, 1), self.capacity).ravel()
        self._rebuild_summaries()

    def refresh(self, X_new: np.ndarray, rate: float, heldout_loss: np.ndarray, heldout_resid: np.ndarray) -> None:
        """Move the reference toward the distribution the model is now adapted to."""
        rate = float(np.clip(rate, 0.0, 1.0))
        X_new = np.asarray(X_new)[:, : self.n_cols]
        k_new = min(len(X_new), int(round(rate * self.capacity)))
        k_old = min(len(self.ref_X), self.capacity - k_new)
        self.ref_X = np.concatenate([self._subsample(self.ref_X, k_old), self._subsample(X_new, k_new)])
        if len(heldout_loss) >= 10:
            self.ref_loss = self._subsample(np.asarray(heldout_loss).reshape(-1, 1), self.capacity).ravel()
            self.ref_resid = self._subsample(np.asarray(heldout_resid).reshape(-1, 1), self.capacity).ravel()
        self._rebuild_summaries()

    def nbytes(self) -> int:
        arrays = [self.ref_X, self.ref_loss, self.ref_resid, self.ref_std, self.ref_corr, self.corr_cols]
        arrays += self.bin_edges + self.ref_bin_p
        return int(sum(a.nbytes for a in arrays if a is not None))

    # -------------------------------------------------------------- testing
    def _feature_tests(self, X: np.ndarray) -> List[FeatureShift]:
        out: List[FeatureShift] = []
        ref = self.ref_X
        for j in range(self.n_num):
            r = ref[:, j][~np.isnan(ref[:, j])]
            c = X[:, j][~np.isnan(X[:, j])]
            if len(r) < 20 or len(c) < 20:
                continue
            ks = stats.ks_2samp(r, c, method="asymp")
            edges = self.bin_edges[j]
            cur_p = _smoothed(np.bincount(np.searchsorted(edges, c, side="right"), minlength=len(edges) + 1))
            w = stats.wasserstein_distance(r, c) / self.ref_std[j]
            out.append(FeatureShift(self.names[j], "numeric", "KS", float(ks.statistic), float(ks.pvalue),
                                    psi=_psi(self.ref_bin_p[j], cur_p), wasserstein=float(w)))
        for j in range(self.n_num, self.n_cols):
            kind = "categorical" if j < self.n_num + self.n_cat else "missingness"
            r = ref[:, j].astype(int)
            c = X[:, j].astype(int)
            m = int(max(r.max(initial=0), c.max(initial=0))) + 1
            rc, cc = np.bincount(r, minlength=m), np.bincount(c, minlength=m)
            keep = (rc + cc) > 0
            if keep.sum() < 2:
                continue
            table = np.vstack([rc[keep], cc[keep]])
            chi2, p, _, _ = stats.chi2_contingency(table + 0.5, correction=False)
            out.append(FeatureShift(self.names[j], kind, "chi2", float(chi2), float(p),
                                    psi=_psi(_smoothed(rc[keep]), _smoothed(cc[keep]))))
        if out:
            q = _bh_qvalues(np.array([f.p_value for f in out]))
            for f, qq in zip(out, q):
                f.q_value = float(qq)
                f.drifted = bool(qq < self.alpha and f.psi >= self.psi_min)
        return out

    def _correlation_test(self, X: np.ndarray):
        if self.ref_corr is None:
            return 0.0, 1.0
        C = self._spearman(X[:, self.corr_cols])
        iu = np.triu_indices(len(self.corr_cols), k=1)
        r1 = np.clip(self.ref_corr[iu], -0.999, 0.999)
        r2 = np.clip(C[iu], -0.999, 0.999)
        n1, n2 = len(self.ref_X), len(X)
        z = np.abs(np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
        p = 2 * stats.norm.sf(z)
        k = int(np.argmin(p))
        return float(np.abs(r1 - r2)[k]), float(min(1.0, p[k] * len(p)))

    def _in_support(self, X: np.ndarray, drifted: List[FeatureShift]) -> np.ndarray:
        mask = np.ones(len(X), dtype=bool)
        index = {n: i for i, n in enumerate(self.names)}
        for f in drifted:
            j = index[f.feature]
            if f.kind == "numeric":
                lo, hi = np.nanquantile(self.ref_X[:, j], [0.005, 0.995])
                v = X[:, j]
                mask &= np.isnan(v) | ((v >= lo) & (v <= hi))
            else:
                seen = np.unique(self.ref_X[:, j])
                mask &= np.isin(X[:, j], seen)
        return mask

    def detect(self, X: np.ndarray, y: np.ndarray, pred_pre: np.ndarray, batch_index: int = -1) -> ShiftReport:
        """Test an incoming labelled batch against the reference, using pre-update predictions."""
        X = np.asarray(X)[:, : self.n_cols]
        y = np.asarray(y, dtype=float)
        pred_pre = np.asarray(pred_pre, dtype=float)
        rep = ShiftReport(batch_index=batch_index, n_samples=len(y))
        self.n_batches += 1

        # ---- covariate / marginal and joint shift
        feats = self._feature_tests(X)
        rep.feature_shifts = feats
        drifted = sorted([f for f in feats if f.drifted], key=lambda f: -f.psi)
        rep.affected_features = [f.feature for f in drifted]
        rep.correlation_max_delta, rep.correlation_p_value = self._correlation_test(X)
        rep.correlation_drift = rep.correlation_p_value < self.alpha and rep.correlation_max_delta >= self.corr_min
        rep.covariate_drift = bool(drifted) or rep.correlation_drift
        cov_mag = max([f.psi for f in drifted] + ([rep.correlation_max_delta * 1.25] if rep.correlation_drift else []) + [0.0])
        rep.covariate_drift_score = float(1 - np.exp(-cov_mag / 0.25)) if rep.covariate_drift else 0.0

        # ---- concept shift on pre-update losses, restricted to reference support
        loss = per_row_loss(self.task_type, y, pred_pre)
        resid = y - pred_pre
        ref_mean = max(float(np.mean(self.ref_loss)), 1e-12)
        rep.loss_ratio_all = float(np.mean(loss) / ref_mean)
        support = self._in_support(X, [f for f in drifted if f.kind != "missingness"])
        rep.in_support_fraction = float(support.mean())
        support_ok = support.sum() >= self.min_support
        L = loss[support] if support_ok else loss
        rep.loss_ratio_in_support = float(np.mean(L) / ref_mean)
        rep.concept_p_value = float(stats.mannwhitneyu(L, self.ref_loss, alternative="greater").pvalue)
        concept = rep.concept_p_value < self.alpha_model and rep.loss_ratio_in_support >= 1 + self.loss_min

        # ---- label / prior shift: systematic residual bias
        R = resid[support] if support_ok else resid
        if self.task_type == "regression":
            rep.bias_p_value = float(stats.ttest_ind(R, self.ref_resid, equal_var=False).pvalue)
            rep.bias_share = float((np.mean(R) - np.mean(self.ref_resid)) ** 2 / max(np.mean(R ** 2), 1e-12))
        else:
            p = np.clip(pred_pre[support] if support_ok else pred_pre, 1e-6, 1 - 1e-6)
            yy = y[support] if support_ok else y
            se = np.sqrt(np.sum(p * (1 - p))) / len(p)
            gap = float(np.mean(yy) - np.mean(p))
            rep.bias_p_value = float(2 * stats.norm.sf(abs(gap) / max(se, 1e-12)))
            ybar = np.clip(np.mean(yy), 1e-3, 1 - 1e-3)
            pbar = np.clip(np.mean(p), 1e-3, 1 - 1e-3)
            logit_gap = abs(np.log(ybar / (1 - ybar)) - np.log(pbar / (1 - pbar)))
            rep.bias_share = float(min(1.0, logit_gap))
        rep.prior_shift = bool(rep.bias_p_value < self.alpha_model and rep.bias_share >= 0.5)
        rep.concept_drift = bool(concept and not (rep.prior_shift and rep.bias_share >= 0.6))

        # ---- gradual drift: Page-Hinkley on log loss ratio across batches
        self.cusum = max(0.0, self.cusum + np.log(max(rep.loss_ratio_all, 1e-6)) - np.log(1.05))
        rep.cusum_statistic = float(self.cusum)
        rep.gradual_drift = bool(self.cusum > np.log(1 + 2 * self.loss_min) and not (rep.concept_drift or rep.prior_shift))
        if rep.concept_drift or rep.prior_shift or rep.gradual_drift:
            self.cusum = 0.0

        con_mag = 0.0
        if rep.concept_drift or rep.gradual_drift:
            con_mag = max(0.0, rep.loss_ratio_in_support - 1)
        if rep.prior_shift:  # a label shift needs plasticity even when the loss barely moves
            con_mag = max(con_mag, 0.5 * rep.bias_share)
        rep.concept_drift_score = float(1 - np.exp(-con_mag / 0.5))
        rep.overall_drift_score = float(1 - (1 - rep.covariate_drift_score) * (1 - rep.concept_drift_score))

        label_change = rep.concept_drift or rep.prior_shift
        if rep.covariate_drift and label_change:
            rep.drift_type = "compound"
        elif rep.prior_shift:
            rep.drift_type = "prior"
        elif rep.concept_drift:
            rep.drift_type = "concept"
        elif rep.covariate_drift:
            rep.drift_type = "covariate"
        elif rep.gradual_drift:
            rep.drift_type = "gradual"
        s = rep.overall_drift_score
        rep.severity = "none" if rep.drift_type == "none" else ("low" if s < 0.3 else "medium" if s < 0.6 else "high")

        decisive = [f.q_value for f in drifted]
        if rep.correlation_drift:
            decisive.append(rep.correlation_p_value)
        if label_change:
            decisive.append(min(rep.concept_p_value, rep.bias_p_value))
        size_factor = len(y) / (len(y) + 100.0) * (1.0 if support_ok else 0.6)
        rep.confidence = float(size_factor * ((1 - min(decisive)) if decisive else (1 - self.alpha)))
        return rep
