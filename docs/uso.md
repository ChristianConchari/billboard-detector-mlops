# Uso

Requiere el stack arriba y con champion ([README](../README.md#cómo-empezar)). Todas las llamadas al
gateway llevan la API key de `.env`; para tenerla en la terminal sin mostrarla:

```bash
set -a; . ./.env; set +a
```

## REST (gateway, puerto 8800)

| Ruta | Para qué |
|---|---|
| `POST /v1/detect` | una imagen JPEG/PNG → cajas, scores, versión del modelo y latencia |
| `GET /v1/model` | modelo cargado, umbral calibrado, mAP del gate con IC 95 % y set de evaluación |
| `POST /v1/reload` | recarga inmediata del champion (clave de `GATEWAY_ADMIN_KEYS`); si no, inference lo busca cada 30 s |
| `GET /health` · `GET /livez` | hay modelo cargado · el gateway alcanza a inference |

```bash
curl -H "X-API-Key: $GATEWAY_API_KEYS" -F image=@<imagen>.jpg localhost:8800/v1/detect
```

Cada predicción queda en `s3://data/predictions/date=…/` con cajas, scores, umbral usado y estadísticas
de brillo y contraste. Las imágenes no se guardan. Contrato completo: [contrato.md](contrato.md).

## gRPC y comparación con REST

`inference` es un servicio interno (gateway y consumidor de streaming); para scripts locales está en
`127.0.0.1:50051` y exige el token de `.env`. El benchmark lee la clave y el token solos:

```bash
scripts/gen_proto.sh   # genera billboard_detector/grpc_gen/ (grpcio-tools==1.81.1)
python scripts/bench_rest_vs_grpc.py <imagen>.jpg -n 40 -c 1 2 4
```

## Streaming con Redpanda

El productor simula cámaras: cada video del dataset es una cámara, sus frames salen en orden con el
instante de captura en el encabezado `event_time_ms`. El servicio `stream-consumer` los infiere y deja
métricas por ventana de 15 s (por cámara) en `curated/stream_windows/`, con alerta si el p95 supera 2 s.

```bash
python scripts/camera_producer.py --rate 2                       # cámaras del test revisado
python scripts/camera_producer.py --rate 2 --degrade darken   # de noche: lo detecta drift_monitoring
python scripts/camera_producer.py --late 0.2                     # 20 % de frames desordenados (watermark)
docker logs -f billboard-detector-mlops-stream-consumer-1
```

(Requiere `kafka-python` y `pillow`: `notebooks/requirements.txt` arma el entorno con todo lo necesario.)

## DAGs de Airflow

| DAG | Cuándo | Qué hace |
|---|---|---|
| `register_model` | al publicar pesos nuevos | descarga y verifica, mide el gate, registra con linaje, re-evalúa al champion, decide el alias. Parámetros: `weights_source` (`gh://`, `hf://repo@revisión`, `s3://data/…`), `min_map`, `min_improvement`, `n_bootstrap` |
| `rollback_model` | si el champion nuevo se porta mal | intercambia `champion` y `previous` |
| `drift_monitoring` | cada hora | drift de las predicciones del intervalo contra la referencia; con drift, alerta y la corrida falla |

```bash
docker exec airflow-apiserver airflow dags trigger register_model --conf '{"min_improvement": 0.02}'
docker exec airflow-apiserver airflow dags trigger drift_monitoring --conf '{"window_hours": 2}'
```

## Escalar la inferencia

```bash
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml --profile all up -d --scale inference=2
```

El gateway reparte entre réplicas por DNS (`round_robin` + `max_connection_age`). A igual CPU da ~20 % más
throughput con varios clientes; una réplica con más hilos (`TORCH_NUM_THREADS=8 INFERENCE_CPUS=8`) da
menor latencia ([arquitectura.md](arquitectura.md#resultados-medidos)). En modo escalado el puerto gRPC del host
cambia al recrear: `docker compose port --index 1 inference 50051`.

## Tests y CI

```bash
pip install -r requirements-dev.txt
pytest -q tests          # sin el stack: gateway con stub gRPC, servidor gRPC real, servicio, gate, drift, ventanas
```

El CI (`.github/workflows/ci.yml`) corre los tests, compila los DAGs y valida `docker compose config` con un
`.env` generado.

## Operación

- **Airflow CLI:** `docker exec airflow-apiserver airflow <comando>` (por ejemplo `airflow dags list-import-errors`).
- **MinIO desde scripts locales:** `AWS_ACCESS_KEY_ID=minio`, `AWS_SECRET_ACCESS_KEY=$MINIO_SECRET_ACCESS_KEY`,
  `AWS_ENDPOINT_URL_S3=http://localhost:9000`.
- **Alertas:** `monitoring/alerts/` en MinIO (`drift`, `error` técnico, `stream` por SLO de latencia).
- **Diagnóstico:** un `503` del gateway puede ser "sin champion", "falló la carga" o "pesos alterados" (inference
  no sirve pesos cuyo SHA-256 no coincide con MLflow): `docker logs billboard-detector-mlops-inference-1`.
  Más trampas conocidas en [estado-y-pendientes.md](estado-y-pendientes.md).
