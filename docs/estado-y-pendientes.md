# Estado y pendientes (traspaso entre máquinas)

Última actualización: 2026-10-14. Escrito para retomar el trabajo en otra máquina sin depender de la memoria de la sesión anterior.
El detalle cronológico completo está en la bitácora del vault (`Second brain/10 Projects/MLOps II - VallaVision/Bitácora del proceso.md`).

## Qué funciona (verificado corriendo el stack)

| Pieza | Estado |
|---|---|
| Instalación desde cero siguiendo solo el README | ✅ 2026-10-14, en una máquina sin volúmenes previos |
| DAG `registrar_modelo` con el RT-DETR de `model-v1` | ✅ 6 tareas OK en ~6 min (CPU); mAP 0.817 / AP50 0.931 sobre el `val` revisado → `champion` automático |
| Gateway REST (`/v1/detect`, `/v1/model`, `/v1/reload`, `/health`) | ✅ mismo score y caja que antes de la separación; 415/422/422 igual |
| Servicio `inference` (gRPC `Detect`, `DetectStream`, `GetModel`, `Reload`) | ✅ healthy, carga el champion |
| Streaming simulado + drift (`scripts/stream_client.py`) | ✅ sin drift normal; con `--degrade oscurecer` marca brillo y contraste |
| DAG `monitoreo_drift` | ✅ reporte en `monitoring/drift/` |
| Tests | ✅ 40 (gateway, servicio, gRPC real en puerto efímero, drift) |

## Qué NO está verificado o está abierto

1. ~~mAP de `val` no comparable (0.066)~~ **Resuelto (2026-10-14):** el DAG evalúa sobre `annotations/reviewed/val.json` (el `val`
   revisado del entrenamiento) y mide 0.817, igual al release. Causa confirmada: 138 de las 169 imágenes del `val` del export eran de `train`
   del modelo, y además 102 de sus cajas estaban sin revisar.
2. **El beneficio de escalar `inference` NO está demostrado.** Primer intento con 2 réplicas: el gateway mandó todo a la réplica 1
   (canal gRPC creado con 1 réplica, no re-resolvía el DNS). Se agregó `grpc.max_connection_age_ms` (30 s) en `inference` para forzar la
   re-resolución, **pero no se llegó a re-medir con la corrección**. Hay que volver a probar (ver "Cómo retomar").
3. **Latencia en CPU ~1.4–2 s por imagen.** La mejor medición limpia (máquina libre) fue ~1.4 s; las demás estaban contaminadas por carga externa
   (navegador, otros proyectos). Sin probar ONNX ni cuantización.

## Medidas de la separación gateway / inference

| Métrica | Antes (`fastapi` + `grpc`) | Después | Notas |
|---|---|---|---|
| Imagen de borde (REST) | 2.3 GB | **245 MB** (gateway) | medido con `docker images` |
| Imagen de inferencia | 2.3 GB (compartida) | 2.28 GB | sigue teniendo torch |
| Build del gateway | – | **31 s** | con la base `python:3.12-slim` ya descargada |
| Build de inferencia | – | 244 s | reinstala torch si cambia `requirements.txt` |
| RAM | 848 + 600 MiB = **1448 MiB** (2 copias del modelo) | 40 + 607 MiB = **647 MiB** | −55 %; 1 copia del modelo |
| Throughput 1 réplica | 0.7 req/s (c=1), 0.8 (c=4); p50 1.4 s | ~0.5–0.6 req/s; p50 1.5–2.0 s | segunda medición con la máquina cargada (load ~4): **no concluyente** |
| Throughput 2 réplicas | – | **sin medir** con la corrección | primer intento: 0.34 req/s, todo en la réplica 1 |

Hipótesis a confirmar: el forward no escala con hilos (4 hilos ≈ 8 hilos en una medición), así que 2 réplicas × 4 hilos deberían acercarse al doble de throughput.
Si no mejora, dejar `--scale` documentado como opción sin prometer ganancia; la ganancia de RAM e imagen ya está demostrada.

## Cómo retomar en otra máquina

Los volúmenes de Docker (MinIO, MLflow, Postgres) **son locales**: en una máquina nueva no existen ni el `champion` ni las predicciones ni la referencia de drift.

```bash
git clone <repo> && cd billboard-detector-mlops
# 1. dataset: descomprimir el zip de origen en data/ -> data/billboards-2025-09-14-sur-lp/{annotations,images,manifest.json}
# 2. .env: AIRFLOW_UID=$(id -u)
docker compose --profile all up -d --build            # ~10 min la primera vez (torch)

# 3. registrar el modelo: con mAP 0.817 >= min_map 0.5 queda champion solo
docker exec airflow-apiserver airflow dags unpause registrar_modelo
docker exec airflow-apiserver airflow dags trigger registrar_modelo   # ~6 min en CPU

# 4. comprobar
curl -s localhost:8800/health
curl -F image=@data/billboards-2025-09-14-sur-lp/images/<una>.jpg localhost:8800/v1/detect
```

Si la build falla con `ReadTimeoutError` de PyPI, construir primero `inference` sola (`docker compose --profile all build inference`).

### Retomar la prueba de escalado

```bash
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml --profile all up -d --scale inference=2
# esperar "champion v1 cargado" en los logs de las dos réplicas y unos 30 s más (re-resolución DNS)
docker stats --no-stream                               # durante la prueba, las dos réplicas deben usar CPU
uv run --with grpcio --with requests --with protobuf scripts/bench_rest_vs_grpc.py <imagen.jpg> -n 12 -c 1 2 4
```
- Medir **con la máquina libre** (sin navegador pesado ni otros proyectos) y comparar 1 réplica vs 2 en el mismo rato.
- Leer la fila **REST** (pasa por el gateway y se reparte). La fila gRPC pega a un solo puerto/réplica.
- En modo escalado el puerto gRPC del host cambia al recrear (`docker compose port --index 1 inference 50051`); por eso el puerto fijo 50051 es solo el modo por defecto.

## Pendientes, en orden sugerido

1. Re-medir el escalado con la corrección de `max_connection_age` y cerrar el punto 2 con números (máquina libre).
2. Latencia: con la máquina libre, y si sigue en ~2 s probar ONNX Runtime o cuantización int8.
3. Slides y defensa (sesión 8); confirmar con el docente el trabajo individual (el reglamento pide 2–6).
4. `docs/alcance.md` todavía dice "ViT" y "DETR sustituto".
5. `DetectStream`: un frame inválido aborta todo el lote.
6. GraphQL (opcional en el alcance), seguridad (JWT, rate limiting en el gateway), tests de los DAGs y CI.

Cerrados el 2026-10-14: evaluación comparable y promoción automática; instalación desde cero; restos del template borrados; numeración de MinIO
y MLflow alineada (en una instalación limpia el RT-DETR es `v1` en ambos); `scripts/revision_etiquetas.py` restaurado.

## Trampas conocidas

- El `val` del export de origen **no** sirve para evaluar `model-v1` (split distinto, cajas sin revisar): usar `annotations/reviewed/val.json`.

- Un 503 por "sin champion" y uno por "falló la carga" se ven igual desde afuera: mirar `docker logs billboard-detector-mlops-inference-1`.
- `docker compose up` con servicios renombrados deja huérfanos: usar `--remove-orphans` (ya se hizo con `fastapi` y `grpc`).
- `/usr/bin/time` no existe en la máquina de desarrollo; medir con `date +%s`.
- Los benchmarks escriben en `predictions/` y alteran lo que ve `monitoreo_drift`.
- Dos numeraciones de versión conviven (MinIO `v<N>` y MLflow): `monitoreo_drift` ya usa el tag `weights_uri`.
- Un servicio con `container_name` no se puede escalar (`inference` no lo tiene; `gateway` sí).
- `vallavision/grpc_gen/` no se versiona: lo generan `scripts/gen_proto.sh`, los Dockerfiles y `tests/conftest.py`.
