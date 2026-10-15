"""DAG rollback_model: goes back to the previous champion.

Every time `register_model` promotes a new champion it gives the outgoing one the `previous`
alias. This DAG swaps `champion` and `previous`; the inference service picks the change up
within 30 s (or immediately with `POST /v1/reload`, admin key). Running it again undoes the
rollback.
"""

import os
from datetime import datetime

from airflow.sdk import dag, task

MODEL_NAME = "billboard-detector"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@dag(dag_id="rollback_model", schedule=None, start_date=datetime(2026, 1, 1), catchup=False,
     max_active_runs=1, tags=["billboard-detector", "model"], doc_md=__doc__)
def rollback_model():

    @task
    def swap_aliases():
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        try:
            previous = client.get_model_version_by_alias(MODEL_NAME, "previous")
        except mlflow.exceptions.MlflowException:
            raise ValueError("there is no `previous` alias: nothing to roll back to") from None
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")
        client.set_registered_model_alias(MODEL_NAME, "champion", previous.version)
        client.set_registered_model_alias(MODEL_NAME, "previous", current.version)
        client.set_model_version_tag(MODEL_NAME, previous.version, "promotion_reason",
                                     f"rollback from v{current.version} ({datetime.utcnow():%F %T} UTC)")
        print(f"champion: v{current.version} → v{previous.version}; previous: v{current.version}")

    swap_aliases()


rollback_model()
