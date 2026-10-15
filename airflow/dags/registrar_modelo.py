"""DAG registrar_modelo: pesos → MinIO → evaluación (mAP + IC) → MLflow → alias.

Layout del Data Lake y contrato de salida en docs/contrato.md (§2 y §4). Gate de promoción
en docs/arquitectura.md:

- El gate se mide sobre `eval_annotations` (por defecto el `test` revisado: videos que no
  intervinieron ni en el entrenamiento ni en la calibración del umbral). El `val` revisado
  solo se usa como referencia de drift.
- El champion actual se re-evalúa sobre el mismo set (o se reutiliza una evaluación previa
  con el mismo sha256 del set): candidato y champion se comparan con la misma vara.
- Cada evaluación queda como un run de MLflow (experimento `evaluaciones`) con el set de
  evaluación como artefacto, su sha256, el hash del código y el commit de git.

Es idempotente: si los pesos ya están en MinIO (mismo sha256) se reutiliza su v<N>, y si
esa v<N> ya está registrada en MLflow no se vuelve a registrar (pero la evaluación nueva sí
se guarda y es la que decide la promoción).
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
GIT_DIR = "/opt/airflow/repo-git"
MODEL_NAME = "billboard-detector"
MODELS_PREFIX = f"models/{MODEL_NAME}"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = "registrar_modelo"
EVAL_EXPERIMENT = "evaluaciones"


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
    from vallavision.promotion import index_by_name

    prefix = f"raw/images/{dataset}/"
    return index_by_name(k for k in _list_keys(s3, prefix)
                         if not k.startswith(prefix + "annotations/") and k.endswith((".jpg", ".jpeg", ".png")))


def _code_sha256():
    """Hash del código que evalúa y sirve (este DAG + vallavision/, sin el gRPC generado)."""
    import hashlib

    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        h.update(f.read())
    root = os.path.join(os.path.dirname(__file__), "vallavision")
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("grpc_gen", "__pycache__"))
        for name in sorted(files):
            if name.endswith(".py"):
                h.update(name.encode())
                with open(os.path.join(dirpath, name), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def _lineage():
    from vallavision.integrity import git_commit

    ctx = get_current_context()
    return {"code_sha256": _code_sha256(), "git_commit": git_commit(GIT_DIR) or "desconocido",
            "airflow_run_id": ctx["run_id"]}


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
    if not expected_sha256:
        raise ValueError("weights_sha256 es obligatorio con gh://: un tag de GitHub se puede mover")
    repo, tag, asset = m.groups()
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "asset.zip")
        h = hashlib.sha256()
        with urllib.request.urlopen(url) as resp, open(zip_path, "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected_sha256.lower():
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
    return h.hexdigest()


def _put_eval_set(s3, local_path, dataset):
    """Sube un COCO de anotaciones a curated/eval/ con su sha256 en el nombre (inmutable)."""
    from vallavision.integrity import sha256_file

    sha = sha256_file(local_path)
    name = os.path.splitext(os.path.basename(local_path))[0]
    key = f"curated/eval/{dataset}/{name}-{sha[:12]}.json"
    if key not in _list_keys(s3, key):
        s3.upload_file(local_path, BUCKET, key)
    return key, sha


def _evaluate(s3, weights_uri, expected_sha, eval_key, dataset, n_bootstrap):
    """Inferencia sobre el set de `eval_key` → métricas del gate (mAP, AP50, IC 95 %)."""
    import tempfile
    import time

    import torch
    from vallavision.detector import BillboardDetector
    from vallavision.integrity import sha256_dir
    from vallavision.promotion import bootstrap_map, coco_map

    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "weights")
        _download_prefix(s3, weights_uri[len(f"s3://{BUCKET}/"):], weights)
        if expected_sha and sha256_dir(weights) != expected_sha:
            raise ValueError(f"los pesos de {weights_uri} no coinciden con su sha256 registrado")
        gt = json.loads(s3.get_object(Bucket=BUCKET, Key=eval_key)["Body"].read())
        detector = BillboardDetector(weights)
        keys = _image_keys(s3, dataset)
        billboard_id = next(c["id"] for c in gt["categories"] if c["name"] == "billboard")

        results, latencies = [], []
        for img in gt["images"]:
            body = s3.get_object(Bucket=BUCKET, Key=keys[img["file_name"]])["Body"].read()
            t0 = time.perf_counter()
            # Umbral bajo: mAP necesita la curva completa de precisión-recall
            _, dets = detector.detect(body, score_threshold=0.01)
            latencies.append((time.perf_counter() - t0) * 1000)
            for d in dets:
                x0, y0, x1, y1 = d["box"]
                results.append({"image_id": img["id"], "category_id": billboard_id,
                                "bbox": [x0, y0, x1 - x0, y1 - y0], "score": d["score"]})

    map_5095, map_50 = coco_map(gt, results)
    ci_low, ci_high = bootstrap_map(gt, results, n=n_bootstrap)
    latencies.sort()
    return {
        "gate_map": map_5095,
        "gate_map_50": map_50,
        "gate_map_ci_low": ci_low,
        "gate_map_ci_high": ci_high,
        "eval_images": len(latencies),
        "eval_detections": len(results),
        # Medida en el worker de Airflow, en paralelo con otras tareas: no es la del servicio
        "eval_latency_ms_mean": sum(latencies) / len(latencies),
        "eval_latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))],
    }


def _log_evaluation(metrics, weights, datos, lineage, role):
    """Run de MLflow con las métricas, el set de evaluación como artefacto y el linaje."""
    import tempfile

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EVAL_EXPERIMENT)
    s3 = _s3()
    with mlflow.start_run(run_name=f"{role}-{weights['uri'].rstrip('/').rsplit('/', 1)[1]}") as run:
        mlflow.set_tags({
            "role": role,
            "weights_uri": weights["uri"],
            "weights_dir_sha256": weights["sha256"],
            "eval_set": datos["eval_set"],
            "eval_sha256": datos["eval_sha256"],
            "dataset": datos["dataset"],
            **lineage,
        })
        mlflow.log_params({"eval_annotations": f"s3://{BUCKET}/{datos['eval_key']}",
                           "n_bootstrap": datos["n_bootstrap"]})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, os.path.basename(datos["eval_key"]))
            s3.download_file(BUCKET, datos["eval_key"], path)
            mlflow.log_artifact(path, artifact_path="eval_set")
        return run.info.run_id


@dag(
    dag_id="registrar_modelo",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Dos corridas en paralelo podrían asignarse la misma v<N> o pisarse los alias
    max_active_runs=1,
    tags=["vallavision", "modelo"],
    params={
        "dataset": Param("billboards-2025-09-14-sur-lp", type="string",
                         description="Carpeta en data/ con images/, annotations/ y manifest.json"),
        "weights_source": Param(
            "gh://ChristianConchari/vit-billboard-detector@model-v1/rtdetr-billboard.zip", type="string",
            description="gh://<owner>/<repo>@<tag>/<asset.zip>, hf://<repo>@<revisión> o s3://data/<prefijo> "
                        "con un checkpoint de transformers"),
        "weights_sha256": Param("264ea1bcf879b08c4cbfb0f82585b31ff3331d562a676786047ca0a5a2e820af",
                                type=["string", "null"],
                                description="SHA-256 del zip (obligatorio con gh://)"),
        "eval_annotations": Param(
            "reviewed/test.json", type="string",
            description="COCO del gate, relativo a annotations/ del repo. Por defecto el test revisado: "
                        "no intervino en el entrenamiento ni en la calibración"),
        "reference_annotations": Param(
            "reviewed/val.json", type="string",
            description="COCO cuyas imágenes forman la referencia de drift (relativo a annotations/)"),
        "min_map": Param(0.5, type="number", minimum=0, maximum=1,
                         description="mAP@[.5:.95] mínimo en el set del gate para ser champion"),
        "min_improvement": Param(0.01, type="number", minimum=0, maximum=1,
                                 description="Mejora mínima de mAP sobre el champion actual (mismo set)"),
        "n_bootstrap": Param(200, type="integer", minimum=0,
                             description="Remuestreos para el IC 95 % del mAP (0 = sin IC)"),
    },
    doc_md=__doc__,
)
def registrar_modelo():

    @task
    def subir_dataset():
        """data/<dataset>/ → raw/images/<dataset>/<split>/; sets de gate y referencia → curated/eval/."""
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

        eval_key, eval_sha = _put_eval_set(s3, os.path.join(LOCAL_ANNOTATIONS, params["eval_annotations"]), dataset)
        ref_key, _ = _put_eval_set(s3, os.path.join(LOCAL_ANNOTATIONS, params["reference_annotations"]), dataset)
        _image_keys(s3, dataset)  # falla temprano si hay nombres repetidos entre splits
        print(f"imágenes subidas={uploaded} ya presentes={skipped}; gate={eval_key} referencia={ref_key}")
        return {"dataset": dataset, "eval_key": eval_key, "eval_sha256": eval_sha,
                "eval_set": f"{params['eval_annotations']}@{eval_sha[:12]}",
                "reference_key": ref_key, "n_bootstrap": int(params["n_bootstrap"])}

    @task
    def descargar_pesos():
        """Checkpoint → models/billboard-detector/v<N>/ + metadata.json (v<N> inmutable)."""
        import tempfile

        from botocore.exceptions import ClientError
        from vallavision.integrity import sha256_dir

        params = get_current_context()["params"]
        source = params["weights_source"]
        s3 = _s3()
        source_sha = None
        with tempfile.TemporaryDirectory() as tmp:
            if source.startswith("gh://"):
                source_sha = _download_github_asset(source, params.get("weights_sha256"), tmp)
            elif source.startswith("hf://"):
                import shutil

                from huggingface_hub import snapshot_download

                repo, _, revision = source[len("hf://"):].partition("@")
                if not revision:
                    raise ValueError("hf:// necesita una revisión fija: hf://<repo>@<commit>")
                snapshot_download(repo, revision=revision, local_dir=tmp,
                                  allow_patterns=["*.json", "*.safetensors"])
                # snapshot_download deja su caché en .cache/; no es parte del checkpoint
                shutil.rmtree(os.path.join(tmp, ".cache"), ignore_errors=True)
            elif source.startswith(f"s3://{BUCKET}/"):
                _download_prefix(s3, source[len(f"s3://{BUCKET}/"):], tmp)
            else:
                raise ValueError(f"weights_source no soportado: {source}")
            sha = sha256_dir(tmp)

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

            # Reservar v<N> con una escritura condicional: si otra corrida la tomó, probar la siguiente
            n = max(versions, default=0) + 1
            while True:
                try:
                    s3.put_object(Bucket=BUCKET, Key=f"{MODELS_PREFIX}/v{n}/.lock", Body=b"", IfNoneMatch="*")
                    break
                except ClientError as e:
                    if e.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
                        raise
                    n += 1
            prefix = f"{MODELS_PREFIX}/v{n}"
            for root, _, files in os.walk(tmp):
                for name in files:
                    path = os.path.join(root, name)
                    s3.upload_file(path, BUCKET, f"{prefix}/{os.path.relpath(path, tmp)}")
            meta = {
                "version": n,
                "uri": f"s3://{BUCKET}/{prefix}/",
                "source": source,
                "source_sha256": source_sha,
                "sha256": sha,  # del directorio del checkpoint (sin metadata.json ni .lock)
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/metadata.json",
                          Body=json.dumps(meta, indent=2).encode())
        print(f"pesos subidos a {meta['uri']}")
        return {**meta, "reused": False}

    @task
    def evaluar(pesos, datos):
        """Métricas del gate del candidato → run de MLflow en `evaluaciones`."""
        metrics = _evaluate(_s3(), pesos["uri"], pesos["sha256"], datos["eval_key"],
                            datos["dataset"], datos["n_bootstrap"])
        run_id = _log_evaluation(metrics, pesos, datos, _lineage(), role="candidato")
        print(json.dumps(metrics, indent=2))
        return {"run_id": run_id, "metrics": metrics}

    @task
    def registrar(pesos, datos, evaluacion):
        """Registra `billboard-detector` (una vez por pesos) y le apunta la evaluación actual."""
        import tempfile

        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.tracking import MlflowClient
        from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

        import vallavision.detector
        from vallavision.detector import FALLBACK_THRESHOLD, read_calibrated_threshold

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        def point_to_evaluation(version):
            # La versión registrada apunta a la evaluación más reciente (la que decide hoy)
            client.set_model_version_tag(MODEL_NAME, version, "eval_run_id", evaluacion["run_id"])
            client.set_model_version_tag(MODEL_NAME, version, "eval_set", datos["eval_set"])

        # Idempotencia: misma v<N> de MinIO → misma versión de MLflow
        try:
            for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
                if mv.tags.get("weights_uri") == pesos["uri"]:
                    print(f"{pesos['uri']} ya registrado como versión {mv.version}")
                    point_to_evaluation(mv.version)
                    return {"model_version": mv.version}
        except mlflow.exceptions.MlflowException:
            pass

        import timm
        import torch
        import transformers
        from PIL import Image

        lineage = _lineage()
        mlflow.set_experiment(EXPERIMENT)
        s3 = _s3()
        with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"v{pesos['version']}") as run:
            weights = os.path.join(tmp, "weights")
            _download_prefix(s3, pesos["uri"][len(f"s3://{BUCKET}/"):], weights)
            for junk in ("metadata.json", ".lock"):
                if os.path.exists(os.path.join(weights, junk)):
                    os.remove(os.path.join(weights, junk))

            mlflow.log_params({
                "weights_source": pesos["source"],
                "weights_uri": pesos["uri"],
                "weights_source_sha256": pesos.get("source_sha256") or "",
                "weights_dir_sha256": pesos["sha256"],
                "dataset": datos["dataset"],
            })
            mlflow.set_tags({**lineage, "eval_run_id": evaluacion["run_id"]})
            mlflow.log_metrics({k: v for k, v in evaluacion["metrics"].items() if k.startswith("gate_")})

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
            for k, v in {"weights_uri": pesos["uri"], "weights_dir_sha256": pesos["sha256"],
                         "weights_source_sha256": pesos.get("source_sha256") or "",
                         "dataset": datos["dataset"], "run_id": run.info.run_id, **lineage}.items():
                client.set_model_version_tag(MODEL_NAME, version, k, v)
            point_to_evaluation(version)
        print(f"registrado {MODEL_NAME} versión {version}")
        return {"model_version": version}

    @task
    def evaluar_champion(registro, datos):
        """mAP del champion actual sobre el MISMO set del gate (reutiliza una evaluación previa
        con el mismo sha256 del set si existe; si no, lo evalúa ahora)."""
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        try:
            champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        except mlflow.exceptions.MlflowException:
            print("no hay champion")
            return None
        if champ.version == registro["model_version"]:
            return {"version": champ.version, "same_as_candidate": True}

        uri = champ.tags["weights_uri"]
        exp = client.get_experiment_by_name(EVAL_EXPERIMENT)
        if exp is not None:
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.weights_uri = '{uri}' and tags.eval_sha256 = '{datos['eval_sha256']}'",
                order_by=["attributes.start_time DESC"], max_results=1)
            if runs:
                print(f"champion v{champ.version}: reutilizo la evaluación {runs[0].info.run_id}")
                return {"version": champ.version, "map": runs[0].data.metrics["gate_map"],
                        "run_id": runs[0].info.run_id}

        expected = champ.tags.get("weights_dir_sha256") or champ.tags.get("weights_sha256")
        metrics = _evaluate(_s3(), uri, expected, datos["eval_key"], datos["dataset"], datos["n_bootstrap"])
        run_id = _log_evaluation(metrics, {"uri": uri, "sha256": expected or ""}, datos, _lineage(), role="champion")
        print(f"champion v{champ.version} re-evaluado: mAP {metrics['gate_map']:.4f}")
        return {"version": champ.version, "map": metrics["gate_map"], "run_id": run_id}

    @task
    def promover(registro, evaluacion, champion):
        """champion si supera min_map y al champion actual (mismo set) por min_improvement."""
        import mlflow
        from mlflow.tracking import MlflowClient
        from vallavision.promotion import decide

        params = get_current_context()["params"]
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        version = registro["model_version"]
        candidate = {"map": evaluacion["metrics"]["gate_map"]}

        if champion and champion.get("same_as_candidate"):
            print(f"v{version} ya es champion")
            return "champion"
        alias, reason = decide(candidate, champion, float(params["min_map"]), float(params["min_improvement"]))
        if alias == "champion" and champion:
            # El champion saliente queda como `anterior`: rollback = DAG rollback_modelo
            client.set_registered_model_alias(MODEL_NAME, "anterior", champion["version"])
        client.set_registered_model_alias(MODEL_NAME, alias, version)
        client.set_model_version_tag(MODEL_NAME, version, "promotion_reason", reason)
        ev = evaluacion["metrics"]
        print(f"v{version} → {alias}: {reason}. IC 95 % del mAP: "
              f"[{ev['gate_map_ci_low']:.3f}, {ev['gate_map_ci_high']:.3f}]")
        return alias

    @task
    def referencia(pesos, datos):
        """Features por frame del set de referencia con el umbral calibrado → referencia de drift."""
        import tempfile

        from PIL import Image
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
            gt = json.loads(s3.get_object(Bucket=BUCKET, Key=datos["reference_key"])["Body"].read())
            detector = BillboardDetector(weights)
            keys = _image_keys(s3, datos["dataset"])
            frames = []
            for img in gt["images"]:
                body = s3.get_object(Bucket=BUCKET, Key=keys[img["file_name"]])["Body"].read()
                _, dets = detector.detect(body)  # umbral calibrado, como en producción
                frames.append(drift.frame_features(dets, drift.image_stats(Image.open(io.BytesIO(body)))))
        ref = {**drift.build_reference(frames), "model_version": pesos["version"],
               "annotations": datos["reference_key"], "score_threshold": detector.default_threshold,
               "created_at": datetime.now(timezone.utc).isoformat()}
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(ref).encode())
        print(f"referencia: {ref['n_frames']} frames → s3://{BUCKET}/{key}")
        return key

    datos = subir_dataset()
    pesos = descargar_pesos()
    evaluacion = evaluar(pesos, datos)
    referencia(pesos, datos)
    registro = registrar(pesos, datos, evaluacion)
    promover(registro, evaluacion, evaluar_champion(registro, datos))


registrar_modelo()
