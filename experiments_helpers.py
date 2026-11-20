import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

def calibration_intercept_slope(y_true, y_prob, eps=1e-15):
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    # calibration intercept (free intercept)
    model = LogisticRegression(fit_intercept=True, solver="lbfgs", penalty=None)
    model.fit(logit, y_true)

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    return intercept, slope


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    # clip so that a probability of exactly 1.0 lands in the last bin instead of being dropped
    binids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = binids == b
        if np.any(mask):
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += np.sum(mask) / n * abs(acc - conf)

    return ece

def logistic_recalibrate(p_val, y_val, p_test, eps=1e-15):
    """Fit intercept + slope on the VALIDATION predictions, apply to TEST predictions.
    Monotone (slope > 0), so AUROC and AUPR are unchanged."""
    def logit(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return np.log(p / (1 - p))
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    lr.fit(logit(p_val).reshape(-1, 1), np.asarray(y_val))
    z = lr.intercept_[0] + lr.coef_[0][0] * logit(p_test)
    return 1.0 / (1.0 + np.exp(-z))


def expected_calibration_error_quantile(y_true, y_prob, n_bins=10):
    """ECE with equal-frequency bins. Use after recalibration: at a 5% event rate
    nearly all calibrated probabilities fall in the first equal-width bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = pd.qcut(pd.Series(y_prob).rank(method='first'), n_bins, labels=False)
    df = pd.DataFrame({'y': y_true, 'p': y_prob, 'b': bins.to_numpy()})
    g = df.groupby('b').agg(n=('y', 'size'), y=('y', 'mean'), p=('p', 'mean'))
    return float((g['n'] / g['n'].sum() * (g['y'] - g['p']).abs()).sum())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)