# Contrato del servicio de detección

Define la forma de entrada/salida del detector **una sola vez**, para que REST, GraphQL,
gRPC, streaming y los resultados en MinIO hablen el mismo idioma. El modelo detrás puede
cambiar sin romper a ningún consumidor. Controles de seguridad: [seguridad.md](seguridad.md).

## 1. Modelo

| Campo | Valor |
|---|---|
| Nombre en MLflow Registry | `billboard-detector` |
| Modelo (v1) | RT-DETR ResNet-50 fine-tuned, release [`model-v1`](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v1) (PyTorch, clase única) |
| Métricas del release | test: mAP 0.649 · AP50 0.930 · AP75 0.794 · 19.5 ms (GPU) |
| Métricas del gate | `gate_map`, `gate_map_50` e IC 95 % por bootstrap, medidas por `registrar_modelo` sobre el `test` revisado (ver §5) |
| Umbral | 0.15, calibrado sobre el `val` revisado (`calibration.json` viaja con los pesos) |
| Aliases | `champion` (modelo servido), `challenger` (candidato que no pasó el gate), `anterior` (champion saliente, para rollback) |

> MLflow 3 depreca las *stages* (`Staging`/`Production`); se usan **aliases**.
> `champion` cumple el rol de "Production".

**Etiquetas.** El contrato no fija la lista de clases: `label` es un `string` y el set
válido lo publica el modelo en su metadata (`GET /v1/model`). El modelo actual tiene una
sola clase, `billboard`.

## 2. REST (gateway, puerto 8800)

**Autenticación.** Todas las rutas salvo `/health` y `/livez` exigen el encabezado
`X-API-Key` con una clave de `GATEWAY_API_KEYS`; `/v1/reload` exige una de `GATEWAY_ADMIN_KEYS`.
Límite de tasa por clave: `GATEWAY_RATE_LIMIT_PER_MIN` (600 por defecto, ventana deslizante de 60 s).

### `POST /v1/detect`

Entrada: `multipart/form-data`, con `Content-Length` (el gateway corta antes de leer el cuerpo si
excede 5 MB + 64 KB).

| Campo | Tipo | Regla |
|---|---|---|
| `image` | archivo | JPEG o PNG **reales** (se mira el formato del archivo, no solo el tipo declarado), ≤ 5 MB, lado mayor ≤ 4096 px |
| `score_threshold` | float (query, opcional) | `0.0–1.0`; por defecto el umbral calibrado del modelo (0.15) |

Salida `200`:

```json
{
  "request_id": "6f1c…",
  "model": { "name": "billboard-detector", "version": "1", "alias": "champion" },
  "image": { "width": 1280, "height": 720 },
  "detections": [
    { "label": "billboard", "score": 0.93, "box": [412.0, 88.5, 790.2, 301.7] }
  ],
  "latency_ms": 412.3
}
```

- `box` = `[x_min, y_min, x_max, y_max]` en **píxeles absolutos** de la imagen tal como se ve
  (si trae orientación EXIF, ya rotada; `image` informa el tamaño rotado).
- `detections` ordenadas por `score` descendente; vacía si nada supera el umbral.
- `latency_ms` = tiempo en el servicio de inferencia desde que llega la request: espera de
  turno + decodificado + forward. No incluye el salto del gateway ni la red.

Errores:

| Código | Cuándo |
|---|---|
| `401` | falta `X-API-Key` |
| `403` | clave inválida, o clave no admin en `/v1/reload` |
| `411` | request sin `Content-Length` |
| `413` | imagen > 5 MB (o cuerpo declarado mayor al límite) |
| `415` | tipo declarado o formato real distinto de JPEG/PNG |
| `422` | imagen ilegible, dimensiones fuera de rango (incluidas "bombas de descompresión") o `score_threshold` inválido |
| `429` | límite de tasa; trae `Retry-After` |
| `502` | error no previsto del servicio de inferencia (sin detalles internos) |
| `503` | no hay modelo `champion` cargado o `inference` no responde |
| `504` | `inference` no respondió a tiempo |

### `GET /v1/model`
Nombre, versión, alias, clases, umbral calibrado, métricas del gate (`gate_map`, `gate_map_50`,
`gate_map_ci_low`, `gate_map_ci_high`) y `eval_set` (`reviewed/test.json@<sha256[:12]>`).

### `POST /v1/reload` (admin)
Fuerza la recarga del `champion` (inference también lo chequea cada 30 s).

### `GET /health` · `GET /livez`
`/health`: `200 {"status": "ok", "model_loaded": true}` · `503` si no hay modelo cargado.
`/livez`: `200` si el gateway alcanza a `inference` (tenga o no modelo); es el healthcheck de Docker.

## 3. gRPC (Sesión 3, servicio `inference`)

Solo en la red interna de Docker (publicado en `127.0.0.1:50051` para los scripts locales) y con
token: metadata `authorization: Bearer <INFERENCE_TOKEN>`; sin token → `UNAUTHENTICATED`.
La imagen viaja como `bytes` (sin base64). `.proto` en `proto/detector.proto`
(`scripts/gen_proto.sh` genera el código). Mensajes de hasta 8 MB.

| RPC | Qué hace |
|---|---|
| `Detect` | una imagen → sus detecciones |
| `DetectStream` | lote de frames en un mensaje → una respuesta por frame (server-streaming) |
| `DetectLive` | **bidireccional**: el cliente manda frames a medida que los produce y recibe una respuesta por frame, en orden; sin límite de lote |
| `GetModel` | modelo cargado (versión, alias, etiquetas, umbral, métricas del gate, `eval_set`, `weights_uri`); `loaded=false` si no hay champion |
| `Reload` | busca el champion en MLflow; `UNAVAILABLE` si no puede (sin detalles) |

`DetectRequest.frame_id` (opcional) vuelve igual en la respuesta. En `DetectStream` y `DetectLive`
un frame inválido **no corta el stream**: su respuesta trae `error {status, message}` con el código
HTTP del contrato (413/415/422) y el resto sigue.

Errores de `Detect`: 413 → `RESOURCE_EXHAUSTED`, 415/422 → `INVALID_ARGUMENT`, sin champion →
`UNAVAILABLE`, cliente que se fue → `CANCELLED`, inesperado → `INTERNAL` (sin detalles). Si el
cliente cancela o vence su deadline mientras espera turno, el frame no se infiere ni se persiste.

**Topología.** `inference` es el único que carga el modelo y persiste. El gateway REST valida tipo,
tamaño y umbral, reenvía los bytes por gRPC y traduce los errores (`INVALID_ARGUMENT` → 422,
`RESOURCE_EXHAUSTED` → 413, `UNAVAILABLE` → 503, `DEADLINE_EXCEEDED` → 504, otro → 502).
`inference` se escala con réplicas (`--scale inference=N`); el gateway las reparte por DNS con
`round_robin` y `max_connection_age` (30 s) lo obliga a re-resolver.

## 4. Data Lake (MinIO)

Bucket `data`, organizado por zonas. Cada servicio tiene su usuario de MinIO con acceso solo a sus
prefijos ([seguridad.md](seguridad.md)).

```
s3://data/
  raw/images/<dataset>/<split>/…                  imágenes originales (split del export de origen)
  raw/stream/date=YYYY-MM-DD/<request_id>.jpg     muestra de las imágenes servidas (PERSIST_IMAGES_RATIO, 10 %)
  models/billboard-detector/v<N>/…                pesos + metadata.json (sha256 del checkpoint y del zip); v<N> inmutable
  predictions/date=YYYY-MM-DD/<request_id>.json   cada inferencia: §2 + campos de monitoreo (abajo)
  curated/eval/<dataset>/<nombre>-<sha12>.json    sets revisados de gate y referencia (inmutables, sha en el nombre)
  curated/reference/billboard-detector/v<N>.json  features del `val` revisado por versión (referencia de drift)
  curated/predictions/model=v<M>/date=…/<intervalo>.parquet   features por frame compactadas por monitoreo_drift
  curated/stream_windows/date=…/<cámara>-<inicio>.json        métricas por ventana del consumidor de streaming
  curated/relabel/<timestamp>.json                imágenes a reetiquetar cuando hay drift
  monitoring/drift/<timestamp>.json               reportes de monitoreo_drift
  monitoring/alerts/<timestamp>-<tipo>.json       alertas: drift, error técnico, stream (drift o SLO)
```

Campos de monitoreo en `predictions/` (REST y gRPC no los devuelven): `inference_ms` (solo el
forward), `score_threshold` (el efectivo), `threshold_is_default`, `image_stats` (`brightness` y
`contrast` de la luminancia, 0–1), `image_format`, `created_at` e `image_key` si la imagen quedó en
`raw/stream/`.

Bucket `mlflow`: artefactos de MLflow (lo gestiona MLflow, no se escribe a mano).

## 5. DAGs (Airflow)

| DAG | Qué hace |
|---|---|
| `registrar_modelo` | descarga pesos (SHA-256 del zip obligatorio) → `models/…/v<N>/` → evalúa sobre el `test` revisado: mAP, AP50 e IC 95 % → run en el experimento `evaluaciones` con el set como artefacto, su sha256, el hash del código y el commit → registra en MLflow → re-evalúa el champion sobre el mismo set → `champion` si mAP ≥ `min_map` (0.5) y supera al champion por `min_improvement` (0.01); si no, `challenger`. El saliente queda como `anterior` |
| `rollback_modelo` | intercambia `champion` y `anterior` |
| `monitoreo_drift` | horario, sobre `data_interval`: compacta las predicciones del champion **con umbral calibrado** a Parquet → chi² (`n_detections`) y KS (`score_mean`, `brightness`, `contrast`) con corrección de Holm (α=0.01 por ventana) → reporte; con drift: alerta + manifiesto de reetiquetado y la tarea falla con `DriftDetected`. Un error técnico deja una alerta `error`. Con menos de 20 frames no emite veredicto |

## 6. GraphQL (Sesión 2, `POST /graphql` en el gateway)

Solo lectura, con `X-API-Key`. Sin introspección (el esquema se documenta acá) y con límite de
profundidad. `limite` se acota a 100.

```graphql
type Query {
  modelo: Modelo                                  # champion cargado en inference
  versiones: [VersionModelo!]!                    # registro de MLflow, más nueva primero
  predicciones(fecha: String!, limite: Int = 20, scoreMinimo: Float = 0): [Prediccion!]!
}
type Modelo { nombre version alias etiquetas umbral conjuntoEvaluacion metricas { nombre valor } }
type VersionModelo { version aliases weightsUri conjuntoEvaluacion gateMap creada }
type Prediccion { requestId versionModelo creada ancho alto latenciaMs detecciones { etiqueta score caja } }
```

## 7. Streaming con broker (Sesión 4)

Redpanda (API de Kafka), topic `frames` con 3 particiones. Mensaje: clave = id de cámara (todos sus
frames en la misma partición, en orden), valor = JPEG, encabezados `event_time_ms` (instante de
captura) y `frame_id`. El consumidor (`stream-consumer`, grupo `detector`) infiere con `Detect`,
agrupa por cámara en ventanas tumbling de 15 s por event time con watermark (tolerancia 2 s; lo que
llega después se cuenta como tardío) y deja en `curated/stream_windows/` frames, throughput,
latencia p50/p95, detecciones por frame, frames tardíos y drift. Alerta si hay drift o p95 > 2 s.
Semántica at-least-once (commit después de inferir).
