"""DAG registrar_modelo: pesos → MinIO → evaluación (mAP) → MLflow → alias.

Layout del Data Lake y contrato de salida en docs/contrato.md (§2 y §4). Es idempotente:
si los pesos ya están en MinIO (mismo sha256) se reutiliza su v<N>, y si esa v<N> ya
está registrada en MLflow no se vuelve a registrar.
"""

import io
import json
import os
import re
from datetime import datetime, timezone

from airflow.sdk import Param, dag, get_current_context, task

BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
LOCAL_DATA = "/opt/airflow/data"
LOCAL_ANNOTATIONS = "/opt/airflow/annotations"
MODEL_NAME = "billboard-detector"
MODELS_PREFIX = f"models/{MODEL_NAME}"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = "registrar_modelo"


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


def _list_keys(s3, prefix):
    """{key: size} de todos los objetos bajo `prefix`."""
    keys = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys[obj["Key"]] = obj["Size"]
    return keys


def _download_prefix(s3, prefix, dest):
    for key in _list_keys(s3, prefix):
        rel = key[len(prefix):].lstrip("/")
        if not rel:
            continue
        path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(BUCKET, key, path)


def _image_keys(s3, dataset):
    """{file_name: key} de las imágenes del dataset en raw/, sin importar el split del export."""
    prefix = f"raw/images/{dataset}/"
    return {key.rsplit("/", 1)[1]: key for key in _list_keys(s3, prefix)
            if not key.startswith(prefix + "annotations/") and key.endswith((".jpg", ".jpeg", ".png"))}


def _download_github_asset(source, expected_sha256, dest):
    """gh://<owner>/<repo>@<tag>/<asset.zip> → checkpoint descomprimido en `dest`."""
    import hashlib
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    m = re.fullmatch(r"gh://([^/]+/[^/@]+)@([^/]+)/(.+\.zip)", source)
    if not m:
        raise ValueError(f"weights_source gh:// inválido: {source}")
    repo, tag, asset = m.groups()
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "asset.zip")
        h = hashlib.sha256()
        with urllib.request.urlopen(url) as resp, open(zip_path, "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
        if expected_sha256 and h.hexdigest() != expected_sha256.lower():
            raise ValueError(f"SHA-256 de {url} no coincide: {h.hexdigest()} != {expected_sha256}")
        extract = os.path.join(tmp, "x")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract)
        # El zip trae una carpeta raíz (rtdetr-billboard/): el checkpoint es su contenido
        root = extract
        entries = os.listdir(extract)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract, entries[0])):
            root = os.path.join(extract, entries[0])
        for name in os.listdir(root):
            shutil.move(os.path.join(root, name), os.path.join(dest, name))


def _sha256_dir(path):
    import hashlib

    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            h.update(name.encode())
            with open(os.path.join(root, name), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


@dag(
    dag_id="registrar_modelo",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["vallavision", "modelo"],
    params={
        "dataset": Param("billboards-2025-09-14-sur-lp", type="string",
                         description="Carpeta en data/ con images/, annotations/ y manifest.json"),
        "weights_source": Param(
            "gh://ChristianConchari/vit-billboard-detector@model-v1/rtdetr-billboard.zip", type="string",
            description="gh://<owner>/<repo>@<tag>/<asset.zip>, hf://<repo> o s3://data/<prefijo> "
                        "con un checkpoint de transformers"),
        "weights_sha256": Param("264ea1bcf879b08c4cbfb0f82585b31ff3331d562a676786047ca0a5a2e820af",
                                type=["string", "null"],
                                description="SHA-256 esperado del zip (solo gh://); vacío para no verificar"),
        "eval_annotations": Param(
            "reviewed/val.json", type="string",
            description="COCO de evaluación, relativo a annotations/ del repo. Por defecto el val revisado "
                        "del entrenamiento (ver annotations/reviewed/README.md)"),
        "min_map": Param(0.5, type="number", minimum=0, maximum=1,
                         description="mAP@[.5:.95] mínimo para ser champion"),
    },
    doc_md=__doc__,
)
def registrar_modelo():

    @task
    def subir_dataset():
        """data/<dataset>/ → raw/images/<dataset>/<split>/ y el set revisado → curated/eval/<dataset>/."""
        params = get_current_context()["params"]
        dataset = params["dataset"]
        src = os.path.join(LOCAL_DATA, dataset)
        s3 = _s3()
        existing = _list_keys(s3, f"raw/images/{dataset}/")

        uploaded = skipped = 0
        for split in ("train", "val", "test"):
            with open(os.path.join(src, "annotations", f"{split}.json")) as f:
                coco = json.load(f)
            for img in coco["images"]:
                path = os.path.join(src, "images", img["file_name"])
                key = f"raw/images/{dataset}/{split}/{img['file_name']}"
                if existing.get(key) == os.path.getsize(path):
                    skipped += 1
                    continue
                s3.upload_file(path, BUCKET, key)
                uploaded += 1
            s3.upload_file(os.path.join(src, "annotations", f"{split}.json"),
                           BUCKET, f"raw/images/{dataset}/annotations/{split}.json")

        s3.upload_file(os.path.join(src, "manifest.json"), BUCKET, f"raw/images/{dataset}/manifest.json")
        eval_key = f"curated/eval/{dataset}/annotations.json"
        s3.upload_file(os.path.join(LOCAL_ANNOTATIONS, params["eval_annotations"]), BUCKET, eval_key)
        print(f"imágenes subidas={uploaded} ya presentes={skipped}")
        return {"dataset": dataset, "eval_annotations": eval_key}

    @task
    def descargar_pesos():
        """Checkpoint → models/billboard-detector/v<N>/ + metadata.json."""
        import tempfile

        params = get_current_context()["params"]
        source = params["weights_source"]
        s3 = _s3()
        with tempfile.TemporaryDirectory() as tmp:
            if source.startswith("gh://"):
                _download_github_asset(source, params.get("weights_sha256"), tmp)
            elif source.startswith("hf://"):
                from huggingface_hub import snapshot_download

                snapshot_download(source[len("hf://"):], local_dir=tmp,
                                  allow_patterns=["*.json", "*.safetensors"])
                # snapshot_download deja su caché en .cache/; no es parte del checkpoint
                import shutil

                shutil.rmtree(os.path.join(tmp, ".cache"), ignore_errors=True)
            elif source.startswith(f"s3://{BUCKET}/"):
                _download_prefix(s3, source[len(f"s3://{BUCKET}/"):], tmp)
            else:
                raise ValueError(f"weights_source no soportado: {source}")
            sha = _sha256_dir(tmp)

            versions = {}
            for key in _list_keys(s3, f"{MODELS_PREFIX}/"):
                m = re.match(rf"{MODELS_PREFIX}/v(\d+)/metadata\.json$", key)
                if m:
                    meta = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
                    versions[int(m.group(1))] = meta
            for n, meta in sorted(versions.items()):
                if meta.get("sha256") == sha:
                    print(f"pesos ya presentes como v{n}")
                    return {**meta, "reused": True}

            n = max(versions, default=0) + 1
            prefix = f"{MODELS_PREFIX}/v{n}"
            for root, _, files in os.walk(tmp):
                for name in files:
                    path = os.path.join(root, name)
                    s3.upload_file(path, BUCKET, f"{prefix}/{os.path.relpath(path, tmp)}")
            meta = {
                "version": n,
                "uri": f"s3://{BUCKET}/{prefix}/",
                "source": source,
                "sha256": sha,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/metadata.json",
                          Body=json.dumps(meta, indent=2).encode())
        print(f"pesos subidos a {meta['uri']}")
        return {**meta, "reused": False}

    @task
    def evaluar(pesos, datos):
        """Inferencia sobre el set revisado y COCO mAP con pycocotools (iscrowd=1 se ignora)."""
        import tempfile
        import time

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from vallavision.detector import BillboardDetector

        s3 = _s3()
        prefix = pesos["uri"][len(f"s3://{BUCKET}/"):]
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            _download_prefix(s3, prefix, weights)
            ann_path = os.path.join(tmp, "annotations.json")
            s3.download_file(BUCKET, datos["eval_annotations"], ann_path)

            detector = BillboardDetector(weights)
            gt = COCO(ann_path)
            keys = _image_keys(s3, datos["dataset"])
            billboard_id = next(c["id"] for c in gt.dataset["categories"] if c["name"] == "billboard")

            results, latencies = [], []
            for img in gt.dataset["images"]:
                body = s3.get_object(Bucket=BUCKET, Key=keys[img["file_name"]])["Body"].read()
                t0 = time.perf_counter()
                # Umbral bajo: mAP necesita la curva completa de precisión-recall
                _, dets = detector.detect(body, score_threshold=0.01)
                latencies.append((time.perf_counter() - t0) * 1000)
                for d in dets:
                    x0, y0, x1, y1 = d["box"]
                    results.append({"image_id": img["id"], "category_id": billboard_id,
                                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": d["score"]})

        if results:
            ev = COCOeval(gt, gt.loadRes(results), iouType="bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            map_5095, map_50 = float(ev.stats[0]), float(ev.stats[1])
        else:
            # loadRes no acepta una lista vacía; sin detecciones el mAP es 0
            print("el modelo no produjo ninguna detección de billboard sobre el set de evaluación")
            map_5095 = map_50 = 0.0

        latencies.sort()
        metrics = {
            "map": max(map_5095, 0.0),  # COCOeval devuelve -1 si no hay GT evaluable
            "map_50": max(map_50, 0.0),
            "eval_images": len(latencies),
            "eval_detections": len(results),
            "latency_ms_mean": sum(latencies) / len(latencies),
            "latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))],
        }
        print(json.dumps(metrics, indent=2))
        return metrics

    @task
    def registrar(pesos, datos, metricas):
        """Loguea params/métricas/modelo pyfunc y lo registra como `billboard-detector`."""
        import tempfile

        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.tracking import MlflowClient
        from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

        import vallavision.detector
        from vallavision.detector import FALLBACK_THRESHOLD, read_calibrated_threshold

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        # Idempotencia: misma v<N> de MinIO → misma versión de MLflow
        try:
            for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
                if mv.tags.get("weights_uri") == pesos["uri"]:
                    print(f"{pesos['uri']} ya registrado como versión {mv.version}")
                    return {"model_version": mv.version}
        except mlflow.exceptions.MlflowException:
            pass

        import timm
        import torch
        import transformers
        from PIL import Image

        mlflow.set_experiment(EXPERIMENT)
        s3 = _s3()
        with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"v{pesos['version']}") as run:
            weights = os.path.join(tmp, "weights")
            _download_prefix(s3, pesos["uri"][len(f"s3://{BUCKET}/"):], weights)

            mlflow.log_params({
                "weights_source": pesos["source"],
                "weights_uri": pesos["uri"],
                "weights_sha256": pesos["sha256"],
                "dataset": datos["dataset"],
                "eval_annotations": f"s3://{BUCKET}/{datos['eval_annotations']}",
            })
            mlflow.log_metrics(metricas)

            # Umbral calibrado sobre validación, viaja con el checkpoint
            threshold = read_calibrated_threshold(weights) or FALLBACK_THRESHOLD
            mlflow.log_param("score_threshold", round(threshold, 4))

            signature = ModelSignature(
                inputs=Schema([ColSpec("binary", "image")]),
                params=ParamSchema([ParamSpec("score_threshold", "double", threshold)]),
            )
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=vallavision.detector.DetectorPyfunc(),
                artifacts={"weights": weights},
                code_paths=[os.path.dirname(vallavision.__file__)],
                signature=signature,
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"torch=={torch.__version__.split('+')[0]}",
                    f"transformers=={transformers.__version__}",
                    f"timm=={timm.__version__}",
                    f"pillow=={Image.__version__}",
                ],
                metadata={"labels": ["billboard"], "score_threshold": threshold,
                          "contract": "docs/contrato.md",
                          "weights_uri": pesos["uri"]},
                registered_model_name=MODEL_NAME,
            )
            version = str(info.registered_model_version)
            for k, v in {"weights_uri": pesos["uri"], "weights_sha256": pesos["sha256"],
                         "dataset": datos["dataset"], "run_id": run.info.run_id}.items():
                client.set_model_version_tag(MODEL_NAME, version, k, v)
        print(f"registrado {MODEL_NAME} versión {version}")
        return {"model_version": version}

    @task
    def promover(registro):
        """champion si mAP ≥ min_map y supera al champion actual; si no, challenger."""
        import mlflow
        from mlflow.tracking import MlflowClient

        min_map = float(get_current_context()["params"]["min_map"])
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        version = registro["model_version"]
        mv = client.get_model_version(MODEL_NAME, version)
        candidate = client.get_run(mv.run_id).data.metrics["map"]

        try:
            champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        except mlflow.exceptions.MlflowException:
            champ = None

        if champ is not None and champ.version == version:
            print(f"v{version} ya es champion")
            return "champion"

        champ_map = client.get_run(champ.run_id).data.metrics["map"] if champ else None
        if candidate >= min_map and (champ is None or candidate > champ_map):
            alias = "champion"
        else:
            alias = "challenger"
        client.set_registered_model_alias(MODEL_NAME, alias, version)
        print(f"v{version} (mAP={candidate:.4f}) → {alias}; "
              f"champion previo: {champ.version if champ else '-'} (mAP={champ_map}), min_map={min_map}")
        return alias

    @task
    def referencia(pesos, datos):
        """Features por frame de val con el umbral calibrado → referencia de drift."""
        import tempfile

        from PIL import Image
        from pycocotools.coco import COCO
        from vallavision import drift
        from vallavision.detector import BillboardDetector

        s3 = _s3()
        key = f"curated/reference/{MODEL_NAME}/v{pesos['version']}.json"
        if key in _list_keys(s3, key):
            print(f"referencia {key} ya existe")
            return key
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            _download_prefix(s3, pesos["uri"][len(f"s3://{BUCKET}/"):], weights)
            ann_path = os.path.join(tmp, "annotations.json")
            s3.download_file(BUCKET, datos["eval_annotations"], ann_path)
            detector = BillboardDetector(weights)
            keys = _image_keys(s3, datos["dataset"])
            frames = []
            for img in COCO(ann_path).dataset["images"]:
                body = s3.get_object(Bucket=BUCKET, Key=keys[img["file_name"]])["Body"].read()
                _, dets = detector.detect(body)  # umbral calibrado, como en producción
                frames.append(drift.frame_features(dets, drift.image_stats(Image.open(io.BytesIO(body)))))
        ref = {**drift.build_reference(frames), "model_version": pesos["version"],
               "created_at": datetime.now(timezone.utc).isoformat()}
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(ref).encode())
        print(f"referencia: {ref['n_frames']} frames → s3://{BUCKET}/{key}")
        return key

    datos = subir_dataset()
    pesos = descargar_pesos()
    metricas = evaluar(pesos, datos)
    referencia(pesos, datos)
    promover(registrar(pesos, datos, metricas))


registrar_modelo()
