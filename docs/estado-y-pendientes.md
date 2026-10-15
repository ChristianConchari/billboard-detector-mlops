# Estado y pendientes (traspaso entre máquinas)

Última actualización: 2026-10-15. Escrito para retomar el trabajo en otra máquina sin depender de la memoria de la sesión anterior.
El detalle cronológico está en la bitácora del vault (`Second brain/10 Projects/MLOps II - VallaVision/Bitácora del proceso.md`).

## Qué funciona (verificado en una instalación limpia, 2026-10-15)

| Pieza | Estado |
|---|---|
| Instalación desde cero (`generar_env.sh` + `up --build` + DAG) | ✅ |
| `registrar_modelo`: gate sobre el `test` revisado con IC, champion re-evaluado, linaje en MLflow | ✅ 3,7 min; mAP 0,647 (IC 0,611–0,705), igual al release |
| `rollback_modelo` | ✅ en el código; sin probar en vivo (hace falta un segundo champion) |
| Gateway REST con API key, rate limit y límites de tamaño | ✅ |
| Inference gRPC (interno, unario) con token y verificación de SHA-256 de los pesos | ✅ |
| Redpanda + `stream-consumer` (métricas por ventana de event time, cierre por inactividad) | ✅ |
| `monitoreo_drift`: Parquet, chi²/KS + Holm y alerta | ✅ control negativo sin drift, positivo con drift y alerta |
| MinIO con un usuario por servicio; puertos de administración solo en `127.0.0.1` | ✅ |
| Tests | ✅ 72; CI en verde en GitHub (tests + `docker compose config`) |

## Mediciones de escalado

2026-10-15, i7-12700F, load < 3,5, REST por el gateway, 40 requests por punto (`bench_rest_vs_grpc.py -n 40 -c 1 2 4`).

| Configuración | c=1 | c=2 | c=4 |
|---|---|---|---|
| 1 réplica × 4 hilos | 2,35 req/s · p50 419 ms | 2,48 · 804 ms | 2,42 · 1245 ms |
| 1 réplica × 8 hilos | **3,28 · 291 ms** | 3,12 · 639 ms | 3,35 · 644 ms |
| 1 réplica × 8 hilos, `INFERENCE_CONCURRENCY=2` | 3,04 · 289 ms | 2,96 · 682 ms | 2,86 · 1387 ms |
| 2 réplicas × 4 hilos | 2,27 · 441 ms | **3,98 · 503 ms** | **4,07 · 968 ms** |

A igual CPU (8 hilos), 2 réplicas dan ~20 % más throughput con 2+ clientes y una réplica de 8 hilos da menor
latencia con un cliente. Dos forwards concurrentes en la misma réplica no ayudan (compiten por los mismos núcleos).
El ×1,7 que se reportaba antes comparaba 2 × 4 hilos contra 1 × 4 hilos (el doble de CPU).

## Cómo retomar en otra máquina

Los volúmenes de Docker (MinIO, MLflow, Postgres, Redpanda) **son locales**: en una máquina nueva no existen ni el
`champion` ni las predicciones ni la referencia de drift. Seguir el README (Instalación): dataset en `data/`,
`scripts/generar_env.sh`, `docker compose --profile all up -d --build`, `registrar_modelo`.

`.env` no se versiona: cada máquina genera el suyo. Si se regenera con volúmenes ya creados, Postgres queda con la
contraseña vieja: `docker compose --profile all down -v` y volver a levantar.

## Pendientes, en orden sugerido

1. Slides y defensa (sesión 8); confirmar con el docente el trabajo individual (el reglamento pide 2–6).
2. Probar `rollback_modelo` en vivo (hace falta un segundo champion).
3. Opcionales: referencias de drift por cámara, ONNX/int8, TLS y MLflow con usuarios.

Fuera de alcance por decisión (sin consumidor o sin caso real; ver arquitectura.md): GraphQL, RPCs de streaming,
drift en el consumidor, muestreo de imágenes y reetiquetado, Label Studio, aprendizaje federado.

## Trampas conocidas

- El `val` del export de origen **no** sirve para evaluar `model-v1` (split distinto, cajas sin revisar); el gate usa
  `annotations/reviewed/test.json` y la referencia de drift `annotations/reviewed/val.json`.
- `inference` queda `unhealthy` hasta que exista un champion (su healthcheck exige el modelo cargado); el gateway
  queda healthy si alcanza a inference.
- Un 503 del gateway puede ser "sin champion", "falló la carga" o "pesos alterados": mirar `docker logs billboard-detector-mlops-inference-1`.
- `docker compose up` con servicios renombrados deja huérfanos: usar `--remove-orphans`.
- Los benchmarks y demos escriben en `predictions/` y alteran lo que ve `monitoreo_drift` en ese intervalo.
- En modo escalado el puerto gRPC del host cambia al recrear (`docker compose port --index 1 inference 50051`).
- `vallavision/grpc_gen/` no se versiona: lo generan `scripts/gen_proto.sh`, los Dockerfiles y `tests/conftest.py` (también si el `.proto` es más nuevo).
- Versiones gRPC fijas en todo el repo: `grpcio==1.84.0`, `grpcio-tools==1.81.1`, `protobuf==6.33.6` (mlflow-skinny exige protobuf < 7).
- En Airflow 3 el `on_failure_callback` no siempre recibe la excepción: `monitoreo_drift` distingue drift de error técnico buscando la alerta `drift` del mismo `run_id`.
