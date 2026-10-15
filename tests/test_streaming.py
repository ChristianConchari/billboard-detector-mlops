"""Ventanas tumbling por event time, watermark y eventos tardíos."""

from vallavision.streaming import TumblingWindows


def test_cierra_por_watermark_y_descarta_tardios():
    w = TumblingWindows(size_ms=1000, allowed_lateness_ms=200)
    assert w.add("cam1", 100, "a") == []
    assert w.add("cam1", 900, "b") == []
    # 1150 - 200 = 950 < 1000: la ventana [0, 1000) sigue abierta (llega tarde pero a tiempo)
    assert w.add("cam1", 1150, "c") == []
    assert w.add("cam1", 980, "d") == []
    closed = w.add("cam1", 1250, "e")  # watermark 1050 ≥ 1000 → cierra [0, 1000)
    assert closed == [("cam1", 0, 1000, ["a", "b", "d"])]
    assert w.add("cam1", 500, "tarde") == [] and w.late["cam1"] == 1
    assert w.flush() == [("cam1", 1000, 2000, ["c", "e"])]


def test_cada_camara_tiene_su_watermark():
    w = TumblingWindows(size_ms=1000)
    w.add("cam1", 5000, "x")
    # cam2 va atrasada respecto de cam1, pero sus eventos no son tardíos
    assert w.add("cam2", 100, "y") == []
    assert w.late["cam2"] == 0
    assert w.add("cam2", 1000, "z") == [("cam2", 0, 1000, ["y"])]


def test_reporte_de_ventana_con_alertas(monkeypatch):
    import random

    from vallavision import drift, stream_consumer

    rnd = random.Random(0)

    def ev(brightness, latency):
        return {"latency_ms": latency, "features": {"n_detections": rnd.randint(0, 2), "score_mean": rnd.uniform(0.4, 0.9),
                                                    "brightness": rnd.gauss(brightness, 0.05),
                                                    "contrast": rnd.gauss(0.2, 0.03)}}

    reference = drift.build_reference([ev(0.5, 0)["features"] for _ in range(200)])
    ok = stream_consumer.window_report("cam1", 0, 15000, [ev(0.5, 400) for _ in range(30)], 0, reference)
    assert ok["frames"] == 30 and ok["throughput_fps"] == 2.0 and ok["drift"] is False and ok["alerts"] == []
    bad = stream_consumer.window_report("cam1", 0, 15000, [ev(0.2, 2500) for _ in range(30)], 3, reference)
    assert bad["late_frames"] == 3 and "brightness" in bad["drift_features"]
    assert any("drift" in a for a in bad["alerts"]) and any("p95" in a for a in bad["alerts"])
