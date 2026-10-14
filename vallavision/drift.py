"""Métricas por frame, resumen por ventana y detección de drift (docs/contrato.md §5).

Cada frame se resume en 4 features. El drift se decide comparando la distribución de
cada feature en la ventana contra la de referencia (el split `val`, inferido con el
mismo modelo) con un test de Kolmogorov-Smirnov de dos muestras.
"""

import math

FEATURES = ("n_detections", "score_mean", "brightness", "contrast")
MIN_FRAMES = 20  # con menos frames el KS no distingue nada


def image_stats(image):
    """Brillo y contraste (media y desvío de la luminancia, 0–1) de una imagen PIL."""
    import numpy as np

    gray = image.convert("L")
    gray.thumbnail((256, 256))
    a = np.asarray(gray, dtype=np.float32) / 255.0
    return {"brightness": round(float(a.mean()), 4), "contrast": round(float(a.std()), 4)}


def frame_features(detections, stats):
    """Features de un frame. `score_mean` es None si no hubo detecciones."""
    scores = [d["score"] for d in detections]
    return {
        "n_detections": len(detections),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "brightness": stats["brightness"],
        "contrast": stats["contrast"],
    }


def build_reference(frames):
    """frames: lista de features por frame → referencia serializable (muestras por feature)."""
    return {"n_frames": len(frames),
            "samples": {f: [x[f] for x in frames if x[f] is not None] for f in FEATURES}}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def compare(reference, frames, alpha=0.01):
    """Compara `frames` (features por frame) contra la referencia.

    Devuelve {"n_frames", "drift", "metrics": {feature: {...}}}; `drift` es True si algún
    feature rechaza H0 (misma distribución) con p < alpha. Con menos de MIN_FRAMES no
    se decide (`drift` None).
    """
    from scipy.stats import ks_2samp

    result = {"n_frames": len(frames), "drift": None, "metrics": {}}
    if len(frames) < MIN_FRAMES:
        return result
    drift = False
    for f in FEATURES:
        cur = [x[f] for x in frames if x[f] is not None]
        ref = reference["samples"][f]
        if len(cur) < MIN_FRAMES or not ref:
            result["metrics"][f] = {"ref_mean": _mean(ref), "cur_mean": _mean(cur), "drift": None}
            continue
        ks = ks_2samp(ref, cur)
        flagged = bool(ks.pvalue < alpha)
        drift = drift or flagged
        result["metrics"][f] = {
            "ref_mean": round(_mean(ref), 4), "cur_mean": round(_mean(cur), 4),
            "ks": round(float(ks.statistic), 4), "p_value": float(f"{ks.pvalue:.3g}"),
            "drift": flagged,
        }
    result["drift"] = drift
    return result


def percentile(values, q):
    """Percentil q (0–100) por rango más cercano; None si no hay valores."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]
