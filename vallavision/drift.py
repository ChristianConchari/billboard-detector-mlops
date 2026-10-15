"""Métricas por frame, resumen por ventana y detección de drift (docs/contrato.md §5).

Cada frame se resume en 4 features. El drift se decide comparando la distribución de
cada feature en la ventana contra la de referencia (el `val` revisado, inferido con el
mismo modelo y el umbral calibrado):

- `n_detections` es un conteo (discreto): test chi² sobre las clases 0, 1, 2 y 3+.
- `score_mean`, `brightness` y `contrast` son continuas: KS de dos muestras.

Son 4 tests por ventana, así que los p-valores se corrigen con Holm-Bonferroni para que
la probabilidad de una falsa alarma por ventana siga siendo ≤ alpha.

Supuesto que NO se cumple con una cámara real: frames independientes. Frames consecutivos
están autocorrelacionados y eso infla los falsos positivos; `thin` permite quedarse con
uno de cada k frames para atenuarlo (docs/arquitectura.md, limitaciones).
"""

import math

FEATURES = ("n_detections", "score_mean", "brightness", "contrast")
DISCRETE = ("n_detections",)
COUNT_BINS = 4  # 0, 1, 2, 3+
MIN_FRAMES = 20  # con menos frames los tests no distinguen nada


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


def usable_for_drift(record, calibrated_threshold=None):
    """Una predicción guardada sirve para el drift solo si usó el umbral calibrado.

    `n_detections` y `score_mean` dependen del umbral: un cliente que pide 0.9 no puede
    disparar una alarma. Registros viejos sin `threshold_is_default` se descartan.
    """
    if "image_stats" not in record or not record.get("threshold_is_default", False):
        return False
    if calibrated_threshold is not None and record.get("score_threshold") is not None:
        return abs(record["score_threshold"] - calibrated_threshold) < 1e-3
    return True


def build_reference(frames):
    """frames: lista de features por frame → referencia serializable (muestras por feature)."""
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
    """Chi² de homogeneidad sobre el histograma de conteos (clases vacías en ambos se quitan)."""
    from scipy.stats import chi2_contingency

    pairs = [(a, b) for a, b in zip(_count_histogram(ref), _count_histogram(cur)) if a + b > 0]
    if len(pairs) < 2:
        return 0.0, 1.0  # una sola clase en ambos: misma distribución
    res = chi2_contingency([[a for a, _ in pairs], [b for _, b in pairs]])
    return float(res.statistic), float(res.pvalue)


def holm(pvalues, alpha):
    """Holm-Bonferroni: {clave: rechaza} controlando el error por familia en `alpha`."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected = {}
    still = True
    for i, (k, p) in enumerate(ordered):
        still = still and p <= alpha / (m - i)
        rejected[k] = still
    return rejected


def thin(frames, k):
    """Uno de cada k frames (atenúa la autocorrelación entre frames consecutivos)."""
    return frames[::k] if k and k > 1 else frames


def compare(reference, frames, alpha=0.01):
    """Compara `frames` (features por frame) contra la referencia.

    Devuelve {"n_frames", "drift", "alpha", "metrics": {feature: {...}}}; `drift` es True si
    algún feature rechaza H0 (misma distribución) tras la corrección de Holm. Con menos de
    MIN_FRAMES no se decide (`drift` None).
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
    """Percentil q (0–100) por rango más cercano; None si no hay valores."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]
