import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vallavision import drift  # noqa: E402


def frames(n, brightness, seed=0):
    rnd = random.Random(seed)
    return [{"n_detections": rnd.randint(0, 3), "score_mean": rnd.uniform(0.4, 0.9),
             "brightness": rnd.gauss(brightness, 0.05), "contrast": rnd.gauss(0.2, 0.03)}
            for _ in range(n)]


def test_image_stats():
    s = drift.image_stats(Image.new("RGB", (40, 40), (255, 255, 255)))
    assert s == {"brightness": 1.0, "contrast": 0.0}


def test_frame_features_sin_detecciones():
    f = drift.frame_features([], {"brightness": 0.5, "contrast": 0.1})
    assert f["n_detections"] == 0 and f["score_mean"] is None


def test_sin_drift_con_misma_distribucion():
    ref = drift.build_reference(frames(200, 0.5, seed=1))
    assert drift.compare(ref, frames(60, 0.5, seed=2))["drift"] is False


def test_drift_por_brillo():
    ref = drift.build_reference(frames(200, 0.5, seed=1))
    res = drift.compare(ref, frames(60, 0.3, seed=2))
    assert res["drift"] is True
    assert res["metrics"]["brightness"]["drift"] is True


def test_pocos_frames_sin_veredicto():
    ref = drift.build_reference(frames(200, 0.5))
    assert drift.compare(ref, frames(5, 0.1))["drift"] is None


def test_percentile():
    assert drift.percentile(list(range(1, 101)), 95) == 95
    assert drift.percentile([], 95) is None
