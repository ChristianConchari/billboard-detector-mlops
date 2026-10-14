# billboard-detector-mlops

**VallaVision**: detector de vallas publicitarias en imágenes de calle, puesto en producción.
TP integrador de *Operaciones de Aprendizaje Automático II* (CEIA · FIUBA). Autor: Christian Conchari.

El foco es la **puesta en producción**, no el modelado: el detector entra como artefacto ya entrenado,
un RT-DETR (ResNet-50) de clase única publicado en el release
[`model-v1`](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v1)
(mAP 0.649 · AP50 0.930 en test, umbral calibrado 0.15).

- [docs/alcance.md](docs/alcance.md): alcance, arquitectura, decisiones, riesgos e hitos
- [docs/contrato.md](docs/contrato.md): contrato de entrada/salida, layout del Data Lake y DAGs
- [docs/arquitectura.md](docs/arquitectura.md): diagrama, decisiones, resultados medidos y limitaciones
- [docs/estado-y-pendientes.md](docs/estado-y-pendientes.md): **estado actual, medidas, pendientes y cómo retomar en otra máquina**

## Servicios

Base: [facundolucianna/amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) (remoto `upstream`).

| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 (airflow / airflow) |
| MLflow | http://localhost:5001 |
| MinIO | http://localhost:9001 (minio / minio123) |
| Gateway REST | http://localhost:8800/docs (sin modelo; reenvía por gRPC) |
| Servicio de inferencia (gRPC) | `localhost:50051` (`proto/detector.proto`); único dueño del modelo, escalable |
| Label Studio (`--profile labeling`) | http://localhost:8090 (admin@vallavision.local / labelstudio) |

Buckets: `data` (Data Lake por zonas, ver contrato §4) y `mlflow` (artefactos de MLflow).

## Dataset y revisión de etiquetas

Dataset `billboards-2025-09-14-sur-lp` (export COCO de origen, clase única `billboard`):
632 imágenes de 26 videos, split por video (train 406 / val 169 / test 57). Las imágenes
no se versionan en git: descomprimir el zip en `data/`, que queda como
`data/billboards-2025-09-14-sur-lp/{annotations,images,manifest.json}`.

El export incluye cajas sin revisar (`verdict: pending`). Para revisarlas con Label Studio:

```bash
docker compose --profile labeling up -d
python scripts/revision_etiquetas.py importar   # proyecto + imágenes con cajas pendientes
# en http://localhost:8090: dejar la caja = confirmar, moverla = ajustar, borrarla = rechazar; Submit
python scripts/revision_etiquetas.py exportar   # -> data/<dataset>/annotations_revisadas/
```

Las cajas rechazadas quedan con `iscrowd: 1` (pycocotools las ignora al evaluar). La
revisión se guarda en el volumen `label_studio_data`, así que se puede exportar varias veces.

## Uso rápido

Con el stack levantado y el modelo registrado (ver [Instalación](#instalación)):

```bash
curl -F image=@data/billboards-2025-09-14-sur-lp/images/<imagen>.jpg http://localhost:8800/v1/detect
```

Sin `champion` el gateway responde `503` y lo busca cada 30 s; `POST /v1/reload` fuerza la recarga.
Cada predicción se guarda en `s3://data/predictions/date=…/<request_id>.json`.

gRPC (`Detect` y `DetectStream`) y comparación contra REST:

```bash
scripts/gen_proto.sh   # genera vallavision/grpc_gen/ (requiere grpcio-tools)
python scripts/bench_rest_vs_grpc.py <imagen.jpg> -n 12 -c 1 2 4   # latencia y req/s con 1, 2 y 4 clientes

# Escalar la inferencia (opcional, beneficio sin demostrar todavía; ver docs/estado-y-pendientes.md):
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml --profile all up -d --scale inference=2
```

Streaming simulado y drift: `scripts/stream_client.py` reproduce una carpeta de imágenes como cámara
(un `Detect` por frame) y reporta por ventana throughput, latencia p50/p95 y detecciones por frame; con
`--reference` también compara contra la referencia (KS) y `--degrade oscurecer` provoca drift para la demo.
El DAG `monitoreo_drift` (horario) hace la misma comparación sobre `predictions/` y falla si hay drift;
la referencia la genera `registrar_modelo` sobre el set de evaluación en `curated/reference/`.

```bash
uv run --with grpcio --with pillow --with numpy --with scipy --with boto3 scripts/stream_client.py \
  data/billboards-2025-09-14-sur-lp/images/ --fps 5 --degrade oscurecer
```

Tests de la API y de drift: `pytest tests` (usan un detector falso, no hace falta el stack).

## Estructura

| Ruta | Qué hay |
|---|---|
| `vallavision/` | Paquete compartido: `detector.py` (RT-DETR + pyfunc de MLflow), `service.py` (carga del champion, validación, persistencia), `grpc_server.py` |
| `vallavision/drift.py` | Features por frame, referencia y test KS |
| `annotations/reviewed/val.json` | Set de evaluación con cajas revisadas (el `val` del entrenamiento del modelo) |
| `airflow/dags/registrar_modelo.py` | DAG de registro y promoción del modelo (genera también la referencia de drift) |
| `airflow/dags/monitoreo_drift.py` | DAG horario: `predictions/` vs referencia → reporte en `monitoring/drift/` |
| `scripts/stream_client.py` | Cámara simulada + métricas por ventana por gRPC |
| `dockerfiles/gateway/` | Gateway REST: FastAPI sin torch, imagen liviana (multi-etapa; contexto de build: raíz del repo) |
| `dockerfiles/inference/` | Servicio de inferencia: torch + modelo, expone gRPC |
| `proto/detector.proto` | Contrato gRPC |

## Estado

| Paso | Estado |
|---|---|
| Stack base (Airflow, MLflow, MinIO, Postgres, FastAPI) levanta healthy | ✅ probado desde cero siguiendo esta guía |
| Contrato de entrada/salida y layout del Data Lake | ✅ |
| DAG `registrar_modelo` (RT-DETR → MinIO → mAP → MLflow → `champion`, referencia de drift) | ✅ promueve solo: mAP 0.817 / AP50 0.931 sobre el `val` revisado (`min_map` 0.5) |
| REST `/v1/detect`, `/v1/model`, `/health` | ✅ probado con el RT-DETR real |
| gRPC `Detect` / `DetectStream` + benchmark vs REST | ✅ probado, con tests |
| Gateway REST (245 MB) + servicio de inferencia separados | ✅ RAM −55 % (647 vs 1448 MiB), mismo resultado que antes, 40 tests |
| Escalado de `inference` con `--scale` | ⏳ sin demostrar: re-medir con `max_connection_age` (ver `docs/estado-y-pendientes.md`) |
| Streaming + drift (`stream_client.py`), DAG `monitoreo_drift` | ✅ sin drift con tráfico normal; con `--degrade oscurecer` marca brillo y contraste |

### Evaluación del modelo

`registrar_modelo` evalúa sobre [`annotations/reviewed/val.json`](annotations/reviewed/README.md): el `val` con cajas
revisadas a mano con el que se entrenó y calibró `model-v1` (95 imágenes, 130 cajas). Mide **mAP 0.817 / AP50 0.931**,
el mismo valor que reporta el release para val, así que la carga y la inferencia del servicio reproducen las del repo del modelo.

No se usa el `val` del export de origen: su split por video no coincide con el del entrenamiento (138 de sus 169
imágenes eran de `train` del modelo) y 102 de sus 169 cajas son pre-etiquetas `pending`. Con ese set el DAG medía mAP 0.066.

## Instalación

### Requisitos

- Docker con Compose v2.
- ~8 GB de RAM libres: el stack usa ~4.7 GB en reposo y el worker de Airflow carga el modelo durante el DAG.
- ~12 GB de disco para las imágenes (Airflow 4.8 GB, MLflow 2.3 GB, inferencia 2.3 GB) y bastante más durante el primer build.
- El zip del dataset `billboards-2025-09-14-sur-lp` (no se versiona en git).

### Pasos

```bash
git clone https://github.com/ChristianConchari/billboard-detector-mlops.git && cd billboard-detector-mlops

# 1. Dataset: queda como data/billboards-2025-09-14-sur-lp/{annotations,images,manifest.json}
mkdir -p data && unzip <ruta>/billboards-2025-09-14-sur-lp-*.zip -d data

# 2. En Linux/macOS, AIRFLOW_UID en .env tiene que ser tu UID (id -u); si no, Airflow deja sus carpetas como root
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env

# 3. Levantar todo (~10 min la primera vez, por torch)
docker compose --profile all up -d --build
docker compose --profile all ps   # todo healthy; airflow-init y minio_create_bucket terminan con Exited (0)

# 4. Registrar el modelo (~6 min en CPU): baja model-v1 del release (verifica SHA-256), lo sube a MinIO,
#    evalúa el mAP, lo registra en MLflow y, si mAP >= 0.5, lo promueve a champion
docker exec airflow-apiserver airflow dags unpause registrar_modelo
docker exec airflow-apiserver airflow dags trigger registrar_modelo

# 5. Comprobar (el gateway busca el champion cada 30 s; POST /v1/reload lo fuerza)
curl -s localhost:8800/health    # {"status":"ok","model_loaded":true}
curl -F image=@data/billboards-2025-09-14-sur-lp/images/<imagen>.jpg localhost:8800/v1/detect
```

Si el build falla con `ReadTimeoutError` de `files.pythonhosted.org`, es la red saturada por bajar todas las
imágenes en paralelo: construir primero la de inferencia (`docker compose --profile all build inference`) y volver a correr el paso 3.

Puertos y credenciales están en `.env` (son de desarrollo). Si el stack corre en otra máquina, reemplazar `localhost` por su IP.

### Apagar

```bash
docker compose --profile all down        # detiene los servicios
docker compose --profile all down -v     # además borra volúmenes: MinIO, MLflow y Postgres (champion, predicciones, referencia)
```

## Operación

- **Airflow CLI:** `docker exec airflow-apiserver airflow <comando>` (por ejemplo `airflow dags list-import-errors`).
  Variables y conexiones persistentes en `airflow/secrets/{variables,connections}.yaml`.
- **MinIO desde scripts locales:** exportar `AWS_ACCESS_KEY_ID=minio`, `AWS_SECRET_ACCESS_KEY=minio123`,
  `AWS_ENDPOINT_URL_S3=http://localhost:9000` y, para MLflow, `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`.
- **Diagnóstico:** un `503` del gateway puede ser "sin champion" o "falló la carga"; mirar
  `docker logs billboard-detector-mlops-inference-1`. Más trampas conocidas en `docs/estado-y-pendientes.md`.
