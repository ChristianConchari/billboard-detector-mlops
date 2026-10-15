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


def test_cierre_por_inactividad():
    w = TumblingWindows(size_ms=1000, allowed_lateness_ms=200)
    w.add("cam1", 100, "a")
    w.add("cam1", 1100, "b")
    w.add("cam2", 100, "c")
    # cam1 dejó de transmitir: sus ventanas se cierran aunque el watermark no haya pasado
    assert w.flush_key("cam1") == [("cam1", 0, 1000, ["a"]), ("cam1", 1000, 2000, ["b"])]
    assert w.add("cam1", 1500, "tarde") == [] and w.late["cam1"] == 1
    assert w.flush() == [("cam2", 0, 1000, ["c"])]


def test_reporte_de_ventana_y_alerta_de_slo():
    from vallavision import stream_consumer

    ok = stream_consumer.window_report("cam1", 0, 15000, [{"latency_ms": 400, "detections": 2}] * 30, 0, slo_ms=2000)
    assert ok["frames"] == 30 and ok["throughput_fps"] == 2.0 and ok["detections_per_frame"] == 2
    assert ok["latency_ms_p95"] == 400 and ok["alerts"] == []
    slow = stream_consumer.window_report("cam1", 0, 15000, [{"latency_ms": 2500, "detections": 1}] * 20, 3, slo_ms=2000)
    assert slow["late_frames"] == 3 and slow["alerts"] == ["latencia p95 2500 ms > SLO 2000 ms"]
