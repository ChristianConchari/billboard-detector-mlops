"""DAG register_model: weights → MinIO → evaluation (mAP + CI) → MLflow → alias.

Data lake layout and output contract in docs/contrato.md (§2 and §4). Promotion gate in
docs/arquitectura.md:

- The gate is measured on `eval_annotations` (by default the reviewed `test` split: videos
  used neither for training nor for calibrating the threshold). The reviewed `val` split
  is only used as the drift reference.
- The current champion is re-evaluated on the same set (or a previous evaluation with the
  same set sha256 is reused): candidate and champion are measured with the same yardstick.
- Every evaluation becomes an MLflow run (experiment `evaluations`) with the evaluation set
  as an artifact, its sha256, the code hash and the git commit.

It is idempotent: if the weights are already in MinIO (same sha256) their v<N> is reused,
and if that v<N> is already registered in MLflow it is not registered again (but the new
evaluation is still logged and is the one that drives the promotion).
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
EXPERIMENT = "register_model"
EVAL_EXPERIMENT = "evaluations"


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


def _list_keys(s3, prefix):
    """{key: size} of every object under `prefix`."""
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
    """{file_name: key} of the dataset images under raw/, whatever split of the export they are in."""
    from billboard_detector.promotion import index_by_name

    prefix = f"raw/images/{dataset}/"
    return index_by_name(k for k in _list_keys(s3, prefix)
                         if not k.startswith(prefix + "annotations/") and k.endswith((".jpg", ".jpeg", ".png")))


def _code_sha256():
    """Hash of the code that evaluates and serves (this DAG + billboard_detector/, minus generated gRPC)."""
    import hashlib

    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        h.update(f.read())
    root = os.path.join(os.path.dirname(__file__), "billboard_detector")
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("grpc_gen", "__pycache__"))
        for name in sorted(files):
            if name.endswith(".py"):
                h.update(name.encode())
                with open(os.path.join(dirpath, name), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def _lineage():
    from billboard_detector.integrity import git_commit

    ctx = get_current_context()
    return {"code_sha256": _code_sha256(), "git_commit": git_commit(GIT_DIR) or "unknown",
            "airflow_run_id": ctx["run_id"]}


def _download_github_asset(source, expected_sha256, dest):
    """gh://<owner>/<repo>@<tag>/<asset.zip> → checkpoint unzipped into `dest`."""
    import hashlib
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    m = re.fullmatch(r"gh://([^/]+/[^/@]+)@([^/]+)/(.+\.zip)", source)
    if not m:
        raise ValueError(f"invalid gh:// weights_source: {source}")
    if not expected_sha256:
        raise ValueError("weights_sha256 is required with gh://: a GitHub tag can be moved")
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
            raise ValueError(f"SHA-256 mismatch for {url}: {h.hexdigest()} != {expected_sha256}")
        extract = os.path.join(tmp, "x")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract)
        # The zip has a root folder (rtdetr-billboard/): the checkpoint is its content
        root = extract
        entries = os.listdir(extract)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract, entries[0])):
            root = os.path.join(extract, entries[0])
        for name in os.listdir(root):
            shutil.move(os.path.join(root, name), os.path.join(dest, name))
    return h.hexdigest()


def _put_eval_set(s3, local_path, dataset):
    """Uploads a COCO annotation file to curated/eval/ with its sha256 in the name (immutable)."""
    from billboard_detector.integrity import sha256_file

    sha = sha256_file(local_path)
    name = os.path.splitext(os.path.basename(local_path))[0]
    key = f"curated/eval/{dataset}/{name}-{sha[:12]}.json"
    if key not in _list_keys(s3, key):
        s3.upload_file(local_path, BUCKET, key)
    return key, sha


def _evaluate(s3, weights_uri, expected_sha, eval_key, dataset, n_bootstrap):
    """Inference over the `eval_key` set → gate metrics (mAP, AP50, 95 % CI)."""
    import tempfile
    import time

    import torch
    from billboard_detector.detector import BillboardDetector
    from billboard_detector.integrity import sha256_dir
    from billboard_detector.promotion import bootstrap_map, coco_map

    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "weights")
        _download_prefix(s3, weights_uri[len(f"s3://{BUCKET}/"):], weights)
        if expected_sha and sha256_dir(weights) != expected_sha:
            raise ValueError(f"the weights at {weights_uri} do not match their registered sha256")
        gt = json.loads(s3.get_object(Bucket=BUCKET, Key=eval_key)["Body"].read())
        detector = BillboardDetector(weights)
        keys = _image_keys(s3, dataset)
        billboard_id = next(c["id"] for c in gt["categories"] if c["name"] == "billboard")

        results, latencies = [], []
        for img in gt["images"]:
            body = s3.get_object(Bucket=BUCKET, Key=keys[img["file_name"]])["Body"].read()
            t0 = time.perf_counter()
            # Low threshold: mAP needs the full precision-recall curve
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
        # Measured in the Airflow worker, in parallel with other tasks: not the service latency
        "eval_latency_ms_mean": sum(latencies) / len(latencies),
        "eval_latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))],
    }


def _log_evaluation(metrics, weights, data, lineage, role):
    """MLflow run with the metrics, the evaluation set as an artifact and the lineage."""
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
            "eval_set": data["eval_set"],
            "eval_sha256": data["eval_sha256"],
            "dataset": data["dataset"],
            **lineage,
        })
        mlflow.log_params({"eval_annotations": f"s3://{BUCKET}/{data['eval_key']}",
                           "n_bootstrap": data["n_bootstrap"]})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, os.path.basename(data["eval_key"]))
            s3.download_file(BUCKET, data["eval_key"], path)
            mlflow.log_artifact(path, artifact_path="eval_set")
        return run.info.run_id


@dag(
    dag_id="register_model",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Two parallel runs could claim the same v<N> or overwrite each other's aliases
    max_active_runs=1,
    tags=["billboard-detector", "model"],
    params={
        "dataset": Param("billboards-2025-09-14-sur-lp", type="string",
                         description="Folder in data/ with images/, annotations/ and manifest.json"),
        "weights_source": Param(
            "gh://ChristianConchari/vit-billboard-detector@model-v1/rtdetr-billboard.zip", type="string",
            description="gh://<owner>/<repo>@<tag>/<asset.zip>, hf://<repo>@<revision> or s3://data/<prefix> "
                        "with a transformers checkpoint"),
        "weights_sha256": Param("264ea1bcf879b08c4cbfb0f82585b31ff3331d562a676786047ca0a5a2e820af",
                                type=["string", "null"],
                                description="SHA-256 of the zip (required with gh://)"),
        "eval_annotations": Param(
            "reviewed/test.json", type="string",
            description="COCO file for the gate, relative to the repo's annotations/. Defaults to the "
                        "reviewed test split: used neither for training nor for calibration"),
        "reference_annotations": Param(
            "reviewed/val.json", type="string",
            description="COCO file whose images form the drift reference (relative to annotations/)"),
        "min_map": Param(0.5, type="number", minimum=0, maximum=1,
                         description="Minimum mAP@[.5:.95] on the gate set to become champion"),
        "min_improvement": Param(0.01, type="number", minimum=0, maximum=1,
                                 description="Minimum mAP improvement over the current champion (same set)"),
        "n_bootstrap": Param(200, type="integer", minimum=0,
                             description="Bootstrap resamples for the mAP 95 % CI (0 = no CI)"),
    },
    doc_md=__doc__,
)
def register_model():

    @task
    def upload_dataset():
        """data/<dataset>/ → raw/images/<dataset>/<split>/; gate and reference sets → curated/eval/."""
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
        _image_keys(s3, dataset)  # fail early if a file name repeats across splits
        print(f"images uploaded={uploaded} already present={skipped}; gate={eval_key} reference={ref_key}")
        return {"dataset": dataset, "eval_key": eval_key, "eval_sha256": eval_sha,
                "eval_set": f"{params['eval_annotations']}@{eval_sha[:12]}",
                "reference_key": ref_key, "n_bootstrap": int(params["n_bootstrap"])}

    @task
    def download_weights():
        """Checkpoint → models/billboard-detector/v<N>/ + metadata.json (v<N> is immutable)."""
        import tempfile

        from billboard_detector.integrity import sha256_dir
        from botocore.exceptions import ClientError

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
                    raise ValueError("hf:// needs a pinned revision: hf://<repo>@<commit>")
                snapshot_download(repo, revision=revision, local_dir=tmp,
                                  allow_patterns=["*.json", "*.safetensors"])
                # snapshot_download leaves its cache in .cache/; it is not part of the checkpoint
                shutil.rmtree(os.path.join(tmp, ".cache"), ignore_errors=True)
            elif source.startswith(f"s3://{BUCKET}/"):
                _download_prefix(s3, source[len(f"s3://{BUCKET}/"):], tmp)
            else:
                raise ValueError(f"unsupported weights_source: {source}")
            sha = sha256_dir(tmp)

            versions = {}
            for key in _list_keys(s3, f"{MODELS_PREFIX}/"):
                m = re.match(rf"{MODELS_PREFIX}/v(\d+)/metadata\.json$", key)
                if m:
                    meta = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
                    versions[int(m.group(1))] = meta
            for n, meta in sorted(versions.items()):
                if meta.get("sha256") == sha:
                    print(f"weights already present as v{n}")
                    return {**meta, "reused": True}

            # Claim v<N> with a conditional write: if another run took it, try the next one
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
                "sha256": sha,  # of the checkpoint directory (without metadata.json and .lock)
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/metadata.json",
                          Body=json.dumps(meta, indent=2).encode())
        print(f"weights uploaded to {meta['uri']}")
        return {**meta, "reused": False}

    @task
    def evaluate(weights, data):
        """Gate metrics of the candidate → MLflow run in `evaluations`."""
        metrics = _evaluate(_s3(), weights["uri"], weights["sha256"], data["eval_key"],
                            data["dataset"], data["n_bootstrap"])
        run_id = _log_evaluation(metrics, weights, data, _lineage(), role="candidate")
        print(json.dumps(metrics, indent=2))
        return {"run_id": run_id, "metrics": metrics}

    @task
    def register(weights, data, evaluation):
        """Registers `billboard-detector` (once per weights) and points it to the current evaluation."""
        import tempfile

        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.tracking import MlflowClient
        from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

        import billboard_detector.detector
        from billboard_detector.detector import FALLBACK_THRESHOLD, read_calibrated_threshold

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        def point_to_evaluation(version):
            # The registered version points to its latest evaluation (the one deciding today)
            client.set_model_version_tag(MODEL_NAME, version, "eval_run_id", evaluation["run_id"])
            client.set_model_version_tag(MODEL_NAME, version, "eval_set", data["eval_set"])

        # Idempotency: same MinIO v<N> → same MLflow version
        try:
            for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
                if mv.tags.get("weights_uri") == weights["uri"]:
                    print(f"{weights['uri']} is already registered as version {mv.version}")
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
        with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"v{weights['version']}") as run:
            weights_dir = os.path.join(tmp, "weights")
            _download_prefix(s3, weights["uri"][len(f"s3://{BUCKET}/"):], weights_dir)
            for junk in ("metadata.json", ".lock"):
                if os.path.exists(os.path.join(weights_dir, junk)):
                    os.remove(os.path.join(weights_dir, junk))

            mlflow.log_params({
                "weights_source": weights["source"],
                "weights_uri": weights["uri"],
                "weights_source_sha256": weights.get("source_sha256") or "",
                "weights_dir_sha256": weights["sha256"],
                "dataset": data["dataset"],
            })
            mlflow.set_tags({**lineage, "eval_run_id": evaluation["run_id"]})
            mlflow.log_metrics({k: v for k, v in evaluation["metrics"].items() if k.startswith("gate_")})

            # Threshold calibrated on validation; it travels with the checkpoint
            threshold = read_calibrated_threshold(weights_dir) or FALLBACK_THRESHOLD
            mlflow.log_param("score_threshold", round(threshold, 4))

            signature = ModelSignature(
                inputs=Schema([ColSpec("binary", "image")]),
                params=ParamSchema([ParamSpec("score_threshold", "double", threshold)]),
            )
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=billboard_detector.detector.DetectorPyfunc(),
                artifacts={"weights": weights_dir},
                code_paths=[os.path.dirname(billboard_detector.__file__)],
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
                          "weights_uri": weights["uri"]},
                registered_model_name=MODEL_NAME,
            )
            version = str(info.registered_model_version)
            for k, v in {"weights_uri": weights["uri"], "weights_dir_sha256": weights["sha256"],
                         "weights_source_sha256": weights.get("source_sha256") or "",
                         "dataset": data["dataset"], "run_id": run.info.run_id, **lineage}.items():
                client.set_model_version_tag(MODEL_NAME, version, k, v)
            point_to_evaluation(version)
        print(f"registered {MODEL_NAME} version {version}")
        return {"model_version": version}

    @task
    def evaluate_champion(registration, data):
        """mAP of the current champion on the SAME gate set (reuses a previous evaluation with
        the same set sha256 if there is one; otherwise evaluates it now)."""
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        try:
            champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        except mlflow.exceptions.MlflowException:
            print("no champion")
            return None
        if champ.version == registration["model_version"]:
            return {"version": champ.version, "same_as_candidate": True}

        uri = champ.tags["weights_uri"]
        exp = client.get_experiment_by_name(EVAL_EXPERIMENT)
        if exp is not None:
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.weights_uri = '{uri}' and tags.eval_sha256 = '{data['eval_sha256']}'",
                order_by=["attributes.start_time DESC"], max_results=1)
            if runs:
                print(f"champion v{champ.version}: reusing evaluation {runs[0].info.run_id}")
                return {"version": champ.version, "map": runs[0].data.metrics["gate_map"],
                        "run_id": runs[0].info.run_id}

        expected = champ.tags.get("weights_dir_sha256") or champ.tags.get("weights_sha256")
        metrics = _evaluate(_s3(), uri, expected, data["eval_key"], data["dataset"], data["n_bootstrap"])
        run_id = _log_evaluation(metrics, {"uri": uri, "sha256": expected or ""}, data, _lineage(), role="champion")
        print(f"champion v{champ.version} re-evaluated: mAP {metrics['gate_map']:.4f}")
        return {"version": champ.version, "map": metrics["gate_map"], "run_id": run_id}

    @task
    def promote(registration, evaluation, champion):
        """champion if it reaches min_map and beats the current champion (same set) by min_improvement."""
        import mlflow
        from billboard_detector.promotion import decide
        from mlflow.tracking import MlflowClient

        params = get_current_context()["params"]
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        version = registration["model_version"]
        candidate = {"map": evaluation["metrics"]["gate_map"]}

        if champion and champion.get("same_as_candidate"):
            print(f"v{version} is already the champion")
            return "champion"
        alias, reason = decide(candidate, champion, float(params["min_map"]), float(params["min_improvement"]))
        if alias == "champion" and champion:
            # The outgoing champion becomes `previous`: rollback = the rollback_model DAG
            client.set_registered_model_alias(MODEL_NAME, "previous", champion["version"])
        client.set_registered_model_alias(MODEL_NAME, alias, version)
        client.set_model_version_tag(MODEL_NAME, version, "promotion_reason", reason)
        ev = evaluation["metrics"]
        print(f"v{version} → {alias}: {reason}. mAP 95 % CI: "
              f"[{ev['gate_map_ci_low']:.3f}, {ev['gate_map_ci_high']:.3f}]")
        return alias

    @task
    def build_reference(weights, data):
        """Per-frame features of the reference set with the calibrated threshold → drift reference."""
        import tempfile

        from billboard_detector import drift
        from billboard_detector.detector import BillboardDetector
        from PIL import Image

        s3 = _s3()
        key = f"curated/reference/{MODEL_NAME}/v{weights['version']}.json"
        if key in _list_keys(s3, key):
            print(f"reference {key} already exists")
            return key
        with tempfile.TemporaryDirectory() as tmp:
            weights_dir = os.path.join(tmp, "weights")
            _download_prefix(s3, weights["uri"][len(f"s3://{BUCKET}/"):], weights_dir)
            gt = json.loads(s3.get_object(Bucket=BUCKET, Key=data["reference_key"])["Body"].read())
            detector = BillboardDetector(weights_dir)
            keys = _image_keys(s3, data["dataset"])
            frames = []
            for img in gt["images"]:
                body = s3.get_object(Bucket=BUCKET, Key=keys[img["file_name"]])["Body"].read()
                _, dets = detector.detect(body)  # calibrated threshold, as in production
                frames.append(drift.frame_features(dets, drift.image_stats(Image.open(io.BytesIO(body)))))
        ref = {**drift.build_reference(frames), "model_version": weights["version"],
               "annotations": data["reference_key"], "score_threshold": detector.default_threshold,
               "created_at": datetime.now(timezone.utc).isoformat()}
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(ref).encode())
        print(f"reference: {ref['n_frames']} frames → s3://{BUCKET}/{key}")
        return key

    data = upload_dataset()
    weights = download_weights()
    evaluation = evaluate(weights, data)
    build_reference(weights, data)
    registration = register(weights, data, evaluation)
    promote(registration, evaluation, evaluate_champion(registration, data))


register_model()
