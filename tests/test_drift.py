"""Per-frame features, chi-squared/KS drift tests and the Holm correction."""

import random

from PIL import Image

from billboard_detector import drift


def frames(n, brightness, seed=0):
    rnd = random.Random(seed)
    return [{"n_detections": rnd.randint(0, 3), "score_mean": rnd.uniform(0.4, 0.9),
             "brightness": rnd.gauss(brightness, 0.05), "contrast": rnd.gauss(0.2, 0.03)}
            for _ in range(n)]


def test_image_stats():
    s = drift.image_stats(Image.new("RGB", (40, 40), (255, 255, 255)))
    assert s == {"brightness": 1.0, "contrast": 0.0}


def test_frame_features_without_detections():
    f = drift.frame_features([], {"brightness": 0.5, "contrast": 0.1})
    assert f["n_detections"] == 0 and f["score_mean"] is None


def test_no_drift_with_the_same_distribution():
    ref = drift.build_reference(frames(200, 0.5, seed=1))
    assert drift.compare(ref, frames(60, 0.5, seed=2))["drift"] is False


def test_brightness_drift():
    ref = drift.build_reference(frames(200, 0.5, seed=1))
    res = drift.compare(ref, frames(60, 0.3, seed=2))
    assert res["drift"] is True
    assert res["metrics"]["brightness"]["drift"] is True


def test_too_few_frames_no_verdict():
    ref = drift.build_reference(frames(200, 0.5))
    assert drift.compare(ref, frames(5, 0.1))["drift"] is None


def test_percentile():
    assert drift.percentile(list(range(1, 101)), 95) == 95
    assert drift.percentile([], 95) is None


def test_count_uses_chi2_and_detects_more_detections():
    ref = drift.build_reference(frames(200, 0.5, seed=1))
    cur = [{**f, "n_detections": 3} for f in frames(60, 0.5, seed=2)]
    res = drift.compare(ref, cur)
    assert res["metrics"]["n_detections"]["test"] == "chi2"
    assert res["metrics"]["n_detections"]["drift"] is True
    assert res["metrics"]["brightness"]["test"] == "ks"


def test_holm_controls_multiple_comparisons():
    # p = 0.004 alone rejects at 0.01, but with 4 tests Holm requires ≤ 0.0025 for the smallest
    assert drift.holm({"a": 0.004, "b": 0.5, "c": 0.6, "d": 0.9}, 0.01) == \
        {"a": False, "b": False, "c": False, "d": False}
    assert drift.holm({"a": 0.001, "b": 0.003, "c": 0.6, "d": 0.9}, 0.01) == \
        {"a": True, "b": True, "c": False, "d": False}


def test_false_alarms_under_h0_are_bounded():
    # 200 windows from the same distribution: with Holm at alpha=0.01 there is almost never an alarm
    ref = drift.build_reference(frames(300, 0.5, seed=1))
    alarms = sum(drift.compare(ref, frames(40, 0.5, seed=s))["drift"] for s in range(100, 300))
    assert alarms <= 6


def test_only_the_calibrated_threshold_counts_for_drift():
    base = {"image_stats": {"brightness": 0.5, "contrast": 0.2}}
    assert drift.usable_for_drift({**base, "threshold_is_default": True, "score_threshold": 0.1502}, 0.1502)
    assert not drift.usable_for_drift({**base, "threshold_is_default": False, "score_threshold": 0.9}, 0.1502)
    assert not drift.usable_for_drift(base)  # old record without the field
    assert not drift.usable_for_drift({"threshold_is_default": True})


def test_thin():
    assert drift.thin(list(range(10)), 3) == [0, 3, 6, 9]
    assert drift.thin([1, 2], 1) == [1, 2]
