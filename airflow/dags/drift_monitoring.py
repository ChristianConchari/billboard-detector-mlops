"""DAG drift_monitoring: the interval's predictions vs the champion's reference.

1. `compact`: reads the predictions of the run's data interval
   (`data_interval_start`–`data_interval_end`, so re-running a past run analyses the same
   window) and writes their features to a single Parquet file under `curated/predictions/`
   (it avoids reading thousands of small JSON files twice and keeps the data out of XCom).
   Only the champion's predictions made with the calibrated threshold count: a client asking
   for another threshold cannot trigger an alarm.
2. `compare`: chi-squared (detection count) and KS (continuous features) with a Holm
   correction against the reference (reviewed `val` split) → report in `monitoring/drift/`.
3. `alert`: on drift, writes a `drift` alert in `monitoring/alerts/` and fails with
   `DriftDetected` (the run turns red in Airflow). A technical error fails with another
   exception and the callback writes an `error` alert. What to do about the drift is a human
   decision: retraining lives in the model's repository, outside this system.

Contract in docs/contrato.md §5.
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import Param, dag, get_current_context, task

BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MODEL_NAME = "billboard-detector"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class DriftDetected(Exception):
    """There is drift: not a technical error (the callback tells them apart)."""


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


def _write_alert(kind, message, **extra):
    now = datetime.now(timezone.utc)
    alert = {"kind": kind, "message": message, "created_at": now.isoformat(), **extra}
    key = f"monitoring/alerts/{now:%Y-%m-%dT%H%M%S}-{kind}.json"
    _s3().put_object(Bucket=BUCKET, Key=key, Body=json.dumps(alert, indent=2).encode())
    print(f"ALERT [{kind}] {message} → s3://{BUCKET}/{key}")
    return key


def _drift_alert_exists(run_id):
    s3 = _s3()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="monitoring/alerts/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("-drift.json"):
                alert = json.loads(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read())
                if alert.get("run_id") == run_id:
                    return True
    return False


def _alert_on_failure(context):
    """`error` alert in monitoring/alerts/ when a task failed for a technical reason.

    Drift does not go through here as an error: `alert` already wrote its `drift` alert with the
    run_id. (In Airflow 3 the callback does not always receive the exception, hence the lookup.)
    """
    ti = context["ti"]
    exc = context.get("exception")
    try:
        if type(exc).__name__ == "DriftDetected" or _drift_alert_exists(ti.run_id):
            return
        message = f"{type(exc).__name__}: {exc}" if exc else f"task {ti.task_id} failed (see its log)"
        _write_alert("error", message, dag_id=ti.dag_id, task_id=ti.task_id, run_id=ti.run_id)
    except Exception as e:  # noqa: BLE001 — the alert must not hide the original error
        print(f"could not store the alert: {e}")


def _window(params):
    """[start, end) of the run: the data interval, or `window_hours` for manual runs."""
    ctx = get_current_context()
    start, end = ctx.get("data_interval_start"), ctx.get("data_interval_end")
    if start is None or end is None or start == end:
        end = ctx.get("run_after") or datetime.now(timezone.utc)
        start = end - timedelta(hours=int(params["window_hours"]))
    return start, end


@dag(
    dag_id="drift_monitoring",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["billboard-detector", "monitoring"],
    default_args={"on_failure_callback": _alert_on_failure},
    params={
        "window_hours": Param(1, type="integer", minimum=1,
                              description="Hours to analyse in manual runs (no data interval)"),
        "alpha": Param(0.01, type="number", minimum=0, maximum=1,
                       description="Family-wise error (Holm) of the 4 tests"),
        "thin": Param(1, type="integer", minimum=1,
                      description="Keep 1 frame out of every k (softens autocorrelation)"),
        "fail_on_drift": Param(True, type="boolean", description="Fail the run when there is drift"),
    },
    doc_md=__doc__,
)
def drift_monitoring():

    @task
    def compact():
        """The interval's predictions/ → curated/predictions/…/<start>.parquet (per-frame features)."""
        import mlflow
        import pandas as pd
        from billboard_detector import drift
        from mlflow.tracking import MlflowClient

        params = get_current_context()["params"]
        start, end = _window(params)
        mlflow.set_tracking_uri(MLFLOW_URI)
        mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        # The reference is stored per MinIO weights version (v<N>), which is not the MLflow
        # version: it is reached through the weights_uri tag
        weights_version = mv.tags["weights_uri"].rstrip("/").rsplit("/v", 1)[1]
        ref_key = f"curated/reference/{MODEL_NAME}/v{weights_version}.json"
        s3 = _s3()
        reference = json.loads(s3.get_object(Bucket=BUCKET, Key=ref_key)["Body"].read())

        rows, skipped = [], 0
        day = start.date()
        while day <= end.date():
            prefix = f"predictions/date={day:%Y-%m-%d}/"
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if not start <= obj["LastModified"] < end:
                        continue
                    rec = json.loads(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read())
                    if rec["model"]["version"] != mv.version or not drift.usable_for_drift(
                            rec, reference.get("score_threshold")):
                        skipped += 1
                        continue
                    rows.append({"request_id": rec["request_id"], "created_at": rec.get("created_at"),
                                 **drift.frame_features(rec["detections"], rec["image_stats"])})
            day += timedelta(days=1)

        key = (f"curated/predictions/model=v{mv.version}/date={start:%Y-%m-%d}/"
               f"{start:%H%M%S}-{end:%H%M%S}.parquet")
        buf = io.BytesIO()
        pd.DataFrame(rows, columns=["request_id", "created_at", *drift.FEATURES]) \
            .sort_values("created_at").to_parquet(buf, index=False)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f"{len(rows)} predictions of v{mv.version} in [{start:%F %T}, {end:%F %T}) "
              f"(skipped {skipped}: other version or non-calibrated threshold) → {key}")
        return {"version": mv.version, "parquet": key, "reference": ref_key, "n_frames": len(rows),
                "start": start.isoformat(), "end": end.isoformat()}

    @task
    def compare(data):
        import pandas as pd
        from billboard_detector import drift

        params = get_current_context()["params"]
        s3 = _s3()
        reference = json.loads(s3.get_object(Bucket=BUCKET, Key=data["reference"])["Body"].read())
        df = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=data["parquet"])["Body"].read()))
        frames = [{f: (None if pd.isna(r[f]) else r[f]) for f in drift.FEATURES} for _, r in df.iterrows()]
        frames = drift.thin(frames, int(params["thin"]))
        result = drift.compare(reference, frames, alpha=float(params["alpha"]))
        report = {"model_version": data["version"], "reference": data["reference"],
                  "window": [data["start"], data["end"]], "features": data["parquet"],
                  "thin": int(params["thin"]), "created_at": datetime.now(timezone.utc).isoformat(), **result}
        out = f"monitoring/drift/{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}.json"
        s3.put_object(Bucket=BUCKET, Key=out, Body=json.dumps(report, indent=2).encode())
        print(json.dumps(report, indent=2))
        print(f"report: s3://{BUCKET}/{out}")
        if result["drift"] is None:
            print(f"too few predictions ({result['n_frames']} < {drift.MIN_FRAMES}): no verdict")
        return {"report": out, "drift": result["drift"],
                "flagged": [f for f, m in result["metrics"].items() if m.get("drift")]}

    @task
    def alert(data, result):
        """On drift: `drift` alert in monitoring/alerts/ and the run fails (visible in Airflow)."""
        if not result["drift"]:
            return None
        reason = f"drift in {', '.join(result['flagged'])}"
        key = _write_alert("drift", reason, report=result["report"], model_version=data["version"],
                           window=[data["start"], data["end"]], run_id=get_current_context()["run_id"])
        if get_current_context()["params"]["fail_on_drift"]:
            raise DriftDetected(f"{reason}; report: {result['report']}")
        return key

    data = compact()
    alert(data, compare(data))


drift_monitoring()
