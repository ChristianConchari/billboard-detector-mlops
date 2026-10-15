# billboard-detector-mlops

**VallaVision**: detector de vallas publicitarias en imágenes de calle, puesto en producción.
TP integrador de *Operaciones de Aprendizaje Automático II* (CEIA · FIUBA). Autor: Christian Conchari.

El foco es la **puesta en producción**, no el modelado: el detector entra como artefacto ya entrenado,
un RT-DETR (ResNet-50) de clase única publicado en el release
[`model-v1`](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v1)
(mAP 0.649 · AP50 0.930 en test, umbral calibrado 0.15). Alrededor: orquestación (Airflow), ciclo de
vida (MLflow), Data Lake (MinIO), serving REST/GraphQL/gRPC, streaming con broker, monitoreo de drift
y seguridad.

| Documento | Qué tiene |
|---|---|
| [docs/arquitectura.md](docs/arquitectura.md) | diagrama, decisiones, gate de promoción, resultados medidos y limitaciones |
| [docs/contrato.md](docs/contrato.md) | contrato REST/GraphQL/gRPC/streaming, layout del Data Lake y DAGs |
| [docs/seguridad.md](docs/seguridad.md) | superficie expuesta y tabla SAIF de riesgos, controles y residuales |
| [docs/estado-y-pendientes.md](docs/estado-y-pendientes.md) | estado, mediciones, pendientes y cómo retomar en otra máquina |
| [docs/alcance.md](docs/alcance.md) | diseño del Hito #1 (histórico) y diferencias con lo implementado |
| [annotations/reviewed/README.md](annotations/reviewed/README.md) | los sets revisados: gate y referencia de drift |

## Servicios

Base: [facundolucianna/amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) (remoto `upstream`).
Solo el gateway escucha fuera de la máquina; el resto está en `127.0.0.1` ([seguridad.md](docs/seguridad.md)).
Usuarios y contraseñas: los de `.env`, que genera `scripts/generar_env.sh`.

| Servicio | Dónde |
|---|---|
| Gateway REST + GraphQL | http://localhost:8800/docs · `POST /graphql` (header `X-API-Key`) |
| Inference (gRPC, único dueño del modelo, escalable) | `127.0.0.1:50051` · `proto/detector.proto` (metadata `authorization: Bearer <INFERENCE_TOKEN>`) |
| Airflow | http://localhost:8080 (`airflow` / `_AIRFLOW_WWW_USER_PASSWORD`) |
| MLflow | http://localhost:5001 |
| MinIO | http://localhost:9001 (`minio` / `MINIO_SECRET_ACCESS_KEY`) |
| Redpanda (Kafka) | `localhost:19092`, topic `frames`; consumidor `stream-consumer` |
| Label Studio (`--profile labeling`) | http://localhost:8090 (`admin@vallavision.local` / `LABEL_STUDIO_PASSWORD`) |

Buckets: `data` (Data Lake por zonas, contrato §4) y `mlflow` (artefactos de MLflow).

## Instalación

### Requisitos

- Docker con Compose v2 y Python 3 (para generar `.env`).
- ~8 GB de RAM libres: el stack usa ~5 GB en reposo y el worker de Airflow carga el modelo en los DAGs.
- ~14 GB de disco para las imágenes (Airflow 4.8 GB, MLflow 2.3 GB, inferencia 2.3 GB, Redpanda 0.6 GB, consumidor 0.6 GB) y bastante más durante el primer build.
- El zip del dataset `billboards-2025-09-14-sur-lp` (no se versiona en git).

### Pasos

```bash
git clone https://github.com/ChristianConchari/billboard-detector-mlops.git && cd billboard-detector-mlops

# 1. Dataset: queda como data/billboards-2025-09-14-sur-lp/{annotations,images,manifest.json}
mkdir -p data && unzip <ruta>/billboards-2025-09-14-sur-lp-*.zip -d data

# 2. .env con secretos aleatorios y AIRFLOW_UID = tu UID (no se versiona)
scripts/generar_env.sh

# 3. Levantar todo (~10 min la primera vez, por torch)
docker compose --profile all up -d --build
docker compose --profile all ps   # healthy salvo inference (unhealthy hasta el paso 4);
                                  # airflow-init y minio_create_bucket terminan con Exited (0)

# 4. Registrar el modelo (~4 min en CPU): baja model-v1 (verifica SHA-256), lo sube a MinIO, mide el
#    mAP con IC sobre el test revisado, lo registra en MLflow con su linaje y, si pasa el gate, champion
docker exec airflow-apiserver airflow dags unpause registrar_modelo
docker exec airflow-apiserver airflow dags trigger registrar_modelo

# 5. Comprobar (inference busca el champion cada 30 s)
set -a; . ./.env; set +a
curl -s localhost:8800/health    # {"status":"ok","model_loaded":true}
curl -H "X-API-Key: $GATEWAY_API_KEYS" -F image=@data/billboards-2025-09-14-sur-lp/images/<imagen>.jpg \
  localhost:8800/v1/detect
```

Si el build falla con `ReadTimeoutError` de `files.pythonhosted.org`, es la red saturada por bajar todas las
imágenes en paralelo: construir primero la de inferencia (`docker compose --profile all build inference`) y volver a correr el paso 3.

Si el stack corre en otra máquina, reemplazar `localhost` por su IP solo para el gateway (lo demás escucha en `127.0.0.1`;
usar un túnel SSH). `GATEWAY_BIND=127.0.0.1` en `.env` deja también el gateway solo local.

### Apagar

```bash
docker compose --profile all down        # detiene los servicios
docker compose --profile all down -v     # además borra volúmenes: MinIO, MLflow, Postgres, Redpanda
```

## Uso

Todas las llamadas al gateway llevan `-H "X-API-Key: $GATEWAY_API_KEYS"` (con `set -a; . ./.env; set +a`).

**REST.** `POST /v1/detect` (imagen), `GET /v1/model` (modelo, umbral, mAP del gate con IC y set de
evaluación), `POST /v1/reload` (clave de `GATEWAY_ADMIN_KEYS`). Cada predicción queda en
`s3://data/predictions/` y el 10 % de las imágenes en `raw/stream/` (para reetiquetar si hay drift).

**GraphQL** (solo lectura, sin introspección; esquema en el contrato §6):

```bash
curl -H "X-API-Key: $GATEWAY_API_KEYS" -H 'Content-Type: application/json' localhost:8800/graphql \
  -d '{"query": "{ modelo { version umbral metricas { nombre valor } } versiones { version aliases gateMap } }"}'
```

**gRPC y benchmark REST vs gRPC** (lee la clave y el token de `.env`):

```bash
uv run --with requests --with grpcio==1.84.0 --with protobuf==6.33.6 scripts/bench_rest_vs_grpc.py <imagen.jpg> -n 40 -c 1 2 4
```
(`scripts/gen_proto.sh` genera `vallavision/grpc_gen/`; requiere `grpcio-tools==1.81.1`.)

**Streaming con broker (Redpanda).** El productor simula cámaras (un video = una cámara) y publica
frames con su event time; el servicio `stream-consumer` los infiere y arma ventanas de 15 s por event
time con watermark → `curated/stream_windows/`, con alertas por drift o p95 > 2 s en `monitoring/alerts/`:

```bash
uv run --with kafka-python --with pillow scripts/camara_productor.py --rate 2              # control negativo
uv run --with kafka-python --with pillow scripts/camara_productor.py --degrade oscurecer   # control positivo
docker logs -f billboard-detector-mlops-stream-consumer-1
```

Por defecto reproduce el `test` revisado: videos que no están en la referencia de drift. Sin broker,
`scripts/stream_client.py` hace lo mismo por gRPC bidireccional (`DetectLive`).

**DAGs.**

| DAG | Para qué |
|---|---|
| `registrar_modelo` | registrar y promover un modelo (gate sobre el `test` revisado, champion re-evaluado sobre el mismo set) |
| `rollback_modelo` | volver al champion anterior (`champion` ↔ `anterior`) |
| `monitoreo_drift` | horario: drift de las predicciones del intervalo; con drift, alerta + manifiesto de reetiquetado |

**Revisión de etiquetas** (Label Studio, sobre el export de origen):

```bash
docker compose --profile labeling up -d
python scripts/revision_etiquetas.py importar   # proyecto + imágenes con cajas pendientes
# en http://localhost:8090: dejar la caja = confirmar, moverla = ajustar, borrarla = rechazar; Submit
python scripts/revision_etiquetas.py exportar   # -> data/<dataset>/annotations_revisadas/
```

**Escalar la inferencia** (el gateway reparte entre réplicas; a igual CPU da ~20 % más throughput, ver arquitectura.md):

```bash
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml --profile all up -d --scale inference=2
```

**Tests** (sin el stack): `pip install -r requirements-dev.txt` y `pytest tests`.
Los corre el CI (`.github/workflows/ci.yml`), junto con `docker compose config` y la compilación de los DAGs.

## Estructura

| Ruta | Qué hay |
|---|---|
| `vallavision/` | Paquete compartido: `detector.py` (RT-DETR + pyfunc), `service.py` (carga verificada del champion, validación, persistencia), `grpc_server.py` (interceptor de token, streams), `validation.py` |
| `vallavision/drift.py` · `promotion.py` · `streaming.py` · `integrity.py` | drift (chi²/KS + Holm), gate y bootstrap, ventanas por event time, hashes y commit |
| `vallavision/stream_consumer.py` | consumidor de Redpanda |
| `airflow/dags/` | `registrar_modelo`, `rollback_modelo`, `monitoreo_drift` |
| `dockerfiles/gateway/` | Gateway REST + GraphQL: FastAPI sin torch (auth, rate limit, límites) |
| `dockerfiles/inference/` · `dockerfiles/stream/` | inferencia (torch + modelo, gRPC) · consumidor de streaming |
| `dockerfiles/minio/init.sh` | buckets y usuarios de MinIO con mínimo privilegio |
| `annotations/reviewed/` | sets revisados versionados (gate y referencia de drift) |
| `scripts/` | `generar_env.sh`, `camara_productor.py`, `stream_client.py`, `bench_rest_vs_grpc.py`, `revision_etiquetas.py`, `gen_proto.sh` |
| `proto/detector.proto` | contrato gRPC |

## Operación

- **Airflow CLI:** `docker exec airflow-apiserver airflow <comando>` (por ejemplo `airflow dags list-import-errors`).
- **MinIO desde scripts locales:** `AWS_ACCESS_KEY_ID=minio`, `AWS_SECRET_ACCESS_KEY=$MINIO_SECRET_ACCESS_KEY`,
  `AWS_ENDPOINT_URL_S3=http://localhost:9000` (y `MLFLOW_S3_ENDPOINT_URL` igual para MLflow).
- **Alertas:** `monitoring/alerts/` en MinIO (`drift`, `error` técnico, `stream`).
- **Diagnóstico:** un `503` del gateway puede ser "sin champion", "falló la carga" o "pesos alterados" (inference no
  sirve pesos cuyo SHA-256 no coincide con MLflow); mirar `docker logs billboard-detector-mlops-inference-1`.
  Más trampas conocidas en `docs/estado-y-pendientes.md`.
