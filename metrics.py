"""
Scale-free evaluation metrics shared by every experiment.

``skill`` is the single primary metric, comparable across tasks:
  * regression:      R^2 = 1 - MSE / Var(y)        (0 = predicting the mean)
  * classification:  Gini = 2 * ROC-AUC - 1        (0 = random ranking)
Both equal 1 for a perfect model. Secondary metrics (RMSE, AUC, log-loss,
Brier) are recorded as well.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def evaluate_predictions(task_type: str, y_true, y_pred, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    if task_type == "regression":
        pred = np.asarray(y_pred, dtype=float)
        mse = float(np.mean((y - pred) ** 2))
        var = float(np.var(y))
        return {"skill": 1.0 - mse / var if var > 1e-12 else 0.0, "rmse": float(np.sqrt(mse)), "r2": 1.0 - mse / var if var > 1e-12 else 0.0}
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    return {
        "skill": 2 * auc - 1,
        "auc": auc,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(np.mean((p >= 0.5) == (y > 0.5))),
    }
