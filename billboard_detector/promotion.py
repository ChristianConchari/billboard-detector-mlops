"""Promotion decision and gate metrics (see the promotion gate in docs/arquitectura.md).

Pure functions (no MLflow, no torch) so they can be tested: the `register_model` DAG uses
them with metrics measured on the same set for both the candidate and the champion.
"""

import contextlib
import io
import random


def decide(candidate, champion, min_map, min_improvement):
    """→ (alias, reason).

    `candidate` and `champion` are {"map": float} measured on the SAME evaluation set;
    `champion` is None when there is none. The candidate becomes champion if it reaches
    `min_map` and beats the current champion by at least `min_improvement` (a tie, or an
    improvement smaller than the set's noise, does not justify replacing the model in production).
    """
    if candidate["map"] < min_map:
        return "challenger", f"mAP {candidate['map']:.4f} < min_map {min_map}"
    if champion is None:
        return "champion", f"no previous champion and mAP {candidate['map']:.4f} ≥ min_map {min_map}"
    delta = candidate["map"] - champion["map"]
    if delta < min_improvement:
        return "challenger", (f"mAP {candidate['map']:.4f} vs champion {champion['map']:.4f} "
                              f"(Δ {delta:+.4f} < {min_improvement})")
    return "champion", f"improves on the champion by Δ {delta:+.4f} ≥ {min_improvement}"


def coco_map(gt_dataset, results):
    """(mAP@[.5:.95], AP50) from COCOeval, silently; 0 when there are no detections or no usable GT."""
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
    # COCOeval returns -1 when there is no usable ground truth
    return max(float(ev.stats[0]), 0.0), max(float(ev.stats[1]), 0.0)


def bootstrap_map(gt_dataset, results, n=200, seed=0, level=0.95):
    """Confidence interval for the mAP by bootstrapping images: (low, high).

    Images are resampled with replacement; repeated images are renumbered because COCOeval
    does not accept duplicate ids. With ~95 images from a few videos the interval is wide:
    that is the real uncertainty of the gate.
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
    """{file_name: key} from S3 keys; fails if two splits contain the same file name."""
    out = {}
    for key in keys:
        name = key.rsplit("/", 1)[1]
        if name in out and out[name] != key:
            raise ValueError(f"duplicate image name in the data lake: {out[name]} and {key}")
        out[name] = key
    return out
