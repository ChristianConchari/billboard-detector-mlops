"""Per-frame features, reference building and drift detection (docs/contrato.md §5).

Every frame is summarised by 4 features. Drift is decided by comparing the distribution
of each feature in the window against the reference (the reviewed `val` split, inferred
with the same model and the calibrated threshold):

- `n_detections` is a count (discrete): chi-squared test over the classes 0, 1, 2 and 3+.
- `score_mean`, `brightness` and `contrast` are continuous: two-sample KS test.

That is 4 tests per window, so p-values are corrected with Holm-Bonferroni to keep the
probability of a false alarm per window at or below alpha.

Assumption that does NOT hold with a real camera: independent frames. Consecutive frames
are autocorrelated, which inflates false positives; `thin` keeps one frame out of every k
to soften it (docs/arquitectura.md, limitations).
"""

import math

FEATURES = ("n_detections", "score_mean", "brightness", "contrast")
DISCRETE = ("n_detections",)
COUNT_BINS = 4  # 0, 1, 2, 3+
MIN_FRAMES = 20  # with fewer frames the tests cannot tell anything apart


def image_stats(image):
    """Brightness and contrast (mean and std of the luminance, 0–1) of a PIL image."""
    import numpy as np

    gray = image.convert("L")
    gray.thumbnail((256, 256))
    a = np.asarray(gray, dtype=np.float32) / 255.0
    return {"brightness": round(float(a.mean()), 4), "contrast": round(float(a.std()), 4)}


def frame_features(detections, stats):
    """Features of one frame. `score_mean` is None when nothing was detected."""
    scores = [d["score"] for d in detections]
    return {
        "n_detections": len(detections),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "brightness": stats["brightness"],
        "contrast": stats["contrast"],
    }


def usable_for_drift(record, calibrated_threshold=None):
    """A stored prediction counts for drift only if it used the calibrated threshold.

    `n_detections` and `score_mean` depend on the threshold: a client asking for 0.9 must
    not be able to trigger an alarm. Old records without `threshold_is_default` are skipped.
    """
    if "image_stats" not in record or not record.get("threshold_is_default", False):
        return False
    if calibrated_threshold is not None and record.get("score_threshold") is not None:
        return abs(record["score_threshold"] - calibrated_threshold) < 1e-3
    return True


def build_reference(frames):
    """frames: list of per-frame features → serialisable reference (samples per feature)."""
    return {"n_frames": len(frames),
            "samples": {f: [x[f] for x in frames if x[f] is not None] for f in FEATURES}}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _count_histogram(xs):
    h = [0] * COUNT_BINS
    for x in xs:
        h[min(int(x), COUNT_BINS - 1)] += 1
    return h


def _chi2_counts(ref, cur):
    """Chi-squared homogeneity test on the count histogram (classes empty in both are dropped)."""
    from scipy.stats import chi2_contingency

    pairs = [(a, b) for a, b in zip(_count_histogram(ref), _count_histogram(cur)) if a + b > 0]
    if len(pairs) < 2:
        return 0.0, 1.0  # a single class in both: same distribution
    res = chi2_contingency([[a for a, _ in pairs], [b for _, b in pairs]])
    return float(res.statistic), float(res.pvalue)


def holm(pvalues, alpha):
    """Holm-Bonferroni: {key: rejected}, controlling the family-wise error at `alpha`."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected = {}
    still = True
    for i, (k, p) in enumerate(ordered):
        still = still and p <= alpha / (m - i)
        rejected[k] = still
    return rejected


def thin(frames, k):
    """One frame out of every k (softens the autocorrelation between consecutive frames)."""
    return frames[::k] if k and k > 1 else frames


def compare(reference, frames, alpha=0.01):
    """Compares `frames` (per-frame features) against the reference.

    Returns {"n_frames", "drift", "alpha", "metrics": {feature: {...}}}; `drift` is True if any
    feature rejects H0 (same distribution) after the Holm correction. With fewer than
    MIN_FRAMES there is no verdict (`drift` is None).
    """
    from scipy.stats import ks_2samp

    result = {"n_frames": len(frames), "drift": None, "alpha": alpha, "metrics": {}}
    if len(frames) < MIN_FRAMES:
        return result
    pvalues = {}
    for f in FEATURES:
        cur = [x[f] for x in frames if x[f] is not None]
        ref = reference["samples"][f]
        if len(cur) < MIN_FRAMES or not ref:
            result["metrics"][f] = {"ref_mean": _mean(ref), "cur_mean": _mean(cur), "drift": None}
            continue
        if f in DISCRETE:
            stat, p = _chi2_counts(ref, cur)
            test = "chi2"
        else:
            ks = ks_2samp(ref, cur)
            stat, p = float(ks.statistic), float(ks.pvalue)
            test = "ks"
        pvalues[f] = p
        result["metrics"][f] = {
            "ref_mean": round(_mean(ref), 4), "cur_mean": round(_mean(cur), 4),
            "test": test, "statistic": round(stat, 4), "p_value": float(f"{p:.3g}"),
        }
    flags = holm(pvalues, alpha)
    for f, flagged in flags.items():
        result["metrics"][f]["drift"] = flagged
    result["drift"] = any(flags.values())
    return result


def percentile(values, q):
    """q-th percentile (0–100) by nearest rank; None when there are no values."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]
