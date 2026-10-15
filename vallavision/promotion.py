"""Decisión de promoción y métricas del gate (docs/arquitectura.md, "Gate de promoción").

Funciones puras (sin MLflow ni torch) para poder testearlas: el DAG `registrar_modelo`
las usa con métricas medidas sobre el mismo set para candidato y champion.
"""

import contextlib
import io
import random


def decide(candidate, champion, min_map, min_improvement):
    """→ (alias, motivo).

    `candidate` y `champion` son {"map": float} medidos sobre el MISMO set de evaluación;
    `champion` es None si no hay. El candidato pasa a champion si supera `min_map` y le
    gana al champion actual por al menos `min_improvement` (un empate o una mejora menor
    que el ruido del set no justifica cambiar el modelo en producción).
    """
    if candidate["map"] < min_map:
        return "challenger", f"mAP {candidate['map']:.4f} < min_map {min_map}"
    if champion is None:
        return "champion", f"sin champion previo y mAP {candidate['map']:.4f} ≥ min_map {min_map}"
    delta = candidate["map"] - champion["map"]
    if delta < min_improvement:
        return "challenger", (f"mAP {candidate['map']:.4f} vs champion {champion['map']:.4f} "
                              f"(Δ {delta:+.4f} < {min_improvement})")
    return "champion", f"mejora Δ {delta:+.4f} ≥ {min_improvement} sobre el champion"


def coco_map(gt_dataset, results):
    """(mAP@[.5:.95], AP50) de COCOeval, silencioso; 0 si no hay detecciones o GT evaluable."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dataset
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(results), iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # COCOeval devuelve -1 si no hay GT evaluable
    return max(float(ev.stats[0]), 0.0), max(float(ev.stats[1]), 0.0)


def bootstrap_map(gt_dataset, results, n=200, seed=0, level=0.95):
    """IC por bootstrap de imágenes para el mAP: (bajo, alto).

    Remuestrea imágenes con reposición; las repetidas se renumeran porque COCOeval no
    admite ids duplicados. Con ~95 imágenes de pocos videos el IC es ancho: es la
    incertidumbre real del gate.
    """
    rng = random.Random(seed)
    images = gt_dataset["images"]
    anns_by_img, res_by_img = {}, {}
    for a in gt_dataset["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    for r in results:
        res_by_img.setdefault(r["image_id"], []).append(r)

    maps = []
    for _ in range(n):
        sample = [rng.choice(images) for _ in images]
        new_images, new_anns, new_res = [], [], []
        for new_id, img in enumerate(sample, start=1):
            new_images.append({**img, "id": new_id})
            for a in anns_by_img.get(img["id"], []):
                new_anns.append({**a, "id": len(new_anns) + 1, "image_id": new_id})
            for r in res_by_img.get(img["id"], []):
                new_res.append({**r, "image_id": new_id})
        ds = {**gt_dataset, "images": new_images, "annotations": new_anns}
        maps.append(coco_map(ds, new_res)[0])
    maps.sort()
    lo = maps[int((1 - level) / 2 * (n - 1))]
    hi = maps[int((1 + level) / 2 * (n - 1))]
    return lo, hi


def index_by_name(keys):
    """{file_name: key} de claves de S3; falla si dos splits tienen el mismo nombre de archivo."""
    out = {}
    for key in keys:
        name = key.rsplit("/", 1)[1]
        if name in out and out[name] != key:
            raise ValueError(f"nombre de imagen repetido en el lake: {out[name]} y {key}")
        out[name] = key
    return out
