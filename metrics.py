"""
metrics.py
==========
Standard saliency-evaluation metrics, following the definitions in
Bylinskii, Judd, Oliva, Torralba & Durand (2018), "What do different
evaluation metrics tell us about saliency models?" (IEEE TPAMI).

These are the same family of metrics (Pearson CC, KLD, SIM) that commercial
platforms cite when validating predictions against human eye-tracking.

pred          : 2-D float array, predicted saliency (any scale)
gt_density    : 2-D float array, human fixation density (blurred fixation map)
gt_fix        : 2-D binary array, 1 where a fixation landed (for NSS / AUC)
"""
import numpy as np

EPS = 1e-12


def _norm_sum(m):
    m = m.astype(np.float64)
    m = m - m.min()
    s = m.sum()
    return m / s if s > 0 else np.full_like(m, 1.0 / m.size)


def cc(pred, gt_density):
    """Pearson linear correlation coefficient (higher = better, max 1)."""
    a = pred.astype(np.float64).ravel()
    b = gt_density.astype(np.float64).ravel()
    a = (a - a.mean()) / (a.std() + EPS)
    b = (b - b.mean()) / (b.std() + EPS)
    return float((a * b).mean())


def sim(pred, gt_density):
    """Histogram intersection between two distributions (higher = better, max 1)."""
    return float(np.minimum(_norm_sum(pred), _norm_sum(gt_density)).sum())


def kld(pred, gt_density):
    """KL divergence of prediction from ground truth (LOWER = better)."""
    p = _norm_sum(pred)
    g = _norm_sum(gt_density)
    return float(np.sum(g * np.log(EPS + g / (p + EPS))))


def nss(pred, gt_fix):
    """Normalized scanpath saliency: mean z-scored prediction at fixations."""
    z = (pred.astype(np.float64) - pred.mean()) / (pred.std() + EPS)
    mask = gt_fix > 0
    return float(z[mask].mean()) if mask.any() else float("nan")


def auc_judd(pred, gt_fix):
    """AUC-Judd: fixated pixels vs all pixels, threshold at each fixated value."""
    p = pred.astype(np.float64).ravel()
    f = (gt_fix > 0).ravel()
    n_fix = int(f.sum())
    if n_fix == 0:
        return float("nan")
    # add tiny jitter so flat maps do not produce degenerate ties
    p = p + np.random.default_rng(0).random(p.shape) * 1e-9
    thr = np.sort(p[f])[::-1]
    n_pix = p.size
    # count of all pixels >= each threshold
    sp = np.sort(p)
    above_all = n_pix - np.searchsorted(sp, thr, side="left")
    tp = (np.arange(1, n_fix + 1)) / n_fix
    fp = (above_all - np.arange(1, n_fix + 1)) / max(n_pix - n_fix, 1)
    tp = np.concatenate([[0], tp, [1]])
    fp = np.concatenate([[0], fp, [1]])
    return float(np.trapezoid(tp, fp)) if hasattr(np, "trapezoid") else float(np.trapz(tp, fp))


def all_metrics(pred, gt_density, gt_fix):
    return {
        "CC": cc(pred, gt_density),
        "SIM": sim(pred, gt_density),
        "KLD": kld(pred, gt_density),
        "NSS": nss(pred, gt_fix),
        "AUC": auc_judd(pred, gt_fix),
    }
