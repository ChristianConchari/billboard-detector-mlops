"""Gate de promoción, mAP con IC por bootstrap e índice de imágenes del lake."""

import pytest

from vallavision import promotion


def test_decide():
    assert promotion.decide({"map": 0.4}, None, 0.5, 0.01)[0] == "challenger"
    assert promotion.decide({"map": 0.6}, None, 0.5, 0.01)[0] == "champion"
    # Contra un champion medido en el mismo set: hace falta superar el margen
    assert promotion.decide({"map": 0.655}, {"map": 0.65}, 0.5, 0.01)[0] == "challenger"
    assert promotion.decide({"map": 0.67}, {"map": 0.65}, 0.5, 0.01)[0] == "champion"
    alias, motivo = promotion.decide({"map": 0.60}, {"map": 0.65}, 0.5, 0.01)
    assert alias == "challenger" and "Δ -0.0500" in motivo


def _dataset(n):
    images = [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100} for i in range(1, n + 1)]
    anns = [{"id": i, "image_id": i, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
            for i in range(1, n + 1)]
    return {"images": images, "annotations": anns, "categories": [{"id": 0, "name": "billboard"}]}


def test_coco_map_perfecto_y_vacio():
    ds = _dataset(10)
    perfect = [{"image_id": i, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9} for i in range(1, 11)]
    assert promotion.coco_map(ds, perfect) == pytest.approx((1.0, 1.0))
    assert promotion.coco_map(ds, []) == (0.0, 0.0)


def test_bootstrap_ic_contiene_el_punto():
    ds = _dataset(20)
    # La mitad de las imágenes con la caja correcta, la otra mitad corrida
    res = [{"image_id": i, "category_id": 0, "bbox": [10, 10, 20, 20] if i % 2 else [60, 60, 20, 20],
            "score": 0.9} for i in range(1, 21)]
    point = promotion.coco_map(ds, res)[0]
    lo, hi = promotion.bootstrap_map(ds, res, n=50)
    assert lo <= point <= hi and hi - lo > 0.05


def test_index_by_name_detecta_colisiones():
    assert promotion.index_by_name(["raw/a/train/x.jpg", "raw/a/val/y.jpg"]) == {
        "x.jpg": "raw/a/train/x.jpg", "y.jpg": "raw/a/val/y.jpg"}
    with pytest.raises(ValueError):
        promotion.index_by_name(["raw/a/train/x.jpg", "raw/a/val/x.jpg"])
