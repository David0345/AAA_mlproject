import numpy as np
from numpy import floating


def logmae_macro(targets: np.ndarray, preds: np.ndarray) -> floating:
    return np.mean(logmae_bycat(targets, preds))


def logmae_bycat(targets: np.ndarray, preds: np.ndarray) -> float:
    targets = np.array(targets)
    preds = np.array(preds)
    targets_log = np.log1p(targets)
    preds_log = np.log1p(preds)
    res = np.abs(targets_log - preds_log)
    return np.mean(res, axis=0)
