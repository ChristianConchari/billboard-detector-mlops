"""DAG rollback_modelo: vuelve al champion anterior.

`registrar_modelo` deja el alias `anterior` en el champion saliente cada vez que promueve
uno nuevo. Este DAG intercambia `champion` y `anterior`; el servicio de inferencia toma el
cambio en ≤ 30 s (o al instante con `POST /v1/reload`, clave admin). Volver a correrlo
deshace el rollback.
"""

import os
from datetime import datetime

from airflow.sdk import dag, task

MODEL_NAME = "billboard-detector"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@dag(dag_id="rollback_modelo", schedule=None, start_date=datetime(2026, 1, 1), catchup=False,
     max_active_runs=1, tags=["vallavision", "modelo"], doc_md=__doc__)
def rollback_modelo():

    @task
    def intercambiar():
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        try:
            previous = client.get_model_version_by_alias(MODEL_NAME, "anterior")
        except mlflow.exceptions.MlflowException:
            raise ValueError("no hay alias `anterior`: no hay a qué volver") from None
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")
        client.set_registered_model_alias(MODEL_NAME, "champion", previous.version)
        client.set_registered_model_alias(MODEL_NAME, "anterior", current.version)
        client.set_model_version_tag(MODEL_NAME, previous.version, "promotion_reason",
                                     f"rollback desde v{current.version} ({datetime.utcnow():%F %T} UTC)")
        print(f"champion: v{current.version} → v{previous.version}; anterior: v{current.version}")

    intercambiar()


rollback_modelo()
