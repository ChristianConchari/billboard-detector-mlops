# Contrato del servicio de detección

Define la forma de entrada/salida del detector **una sola vez**, para que REST, gRPC,
streaming y los resultados en MinIO hablen el mismo idioma. El modelo detrás puede
cambiar sin romper a ningún consumidor.

## 1. Modelo

| Campo | Valor |
|---|---|
| Nombre en MLflow Registry | `billboard-detector` |
| Modelo (v1) | RT-DETR ResNet-50 fine-tuned, release [`model-v1`](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v1) (PyTorch, clase única) |
| Métricas (test) | mAP 0.649 · AP50 0.930 · AP75 0.794 · 19.5 ms |
| Umbral | 0.15, calibrado sobre validación (`calibration.json` viaja con los pesos) |
| Promoción | alias `champion` (modelo servido) y `challenger` (candidato) |

> MLflow 3 depreca las *stages* (`Staging`/`Production`); se usan **aliases**.
> `champion` cumple el rol de "Production" del documento de alcance.

**Etiquetas.** El contrato no fija la lista de clases: `label` es un `string` y el set
válido lo publica el modelo en su metadata (`GET /v1/model`). El modelo actual tiene una
sola clase, `billboard`.

## 2. REST

### `POST /v1/detect`

Entrada: `multipart/form-data`

| Campo | Tipo | Regla |
|---|---|---|
| `image` | archivo | `image/jpeg` o `image/png`, ≤ 5 MB, lado mayor ≤ 4096 px |
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

- `box` = `[x_min, y_min, x_max, y_max]` en **píxeles absolutos** de la imagen original.
- `detections` ordenadas por `score` descendente; vacía si nada supera el umbral.

Errores:

| Código | Cuándo |
|---|---|
| `415` | tipo de archivo no soportado |
| `413` | imagen > 5 MB |
| `422` | imagen ilegible, dimensiones fuera de rango o `score_threshold` inválido |
| `503` | no hay modelo `champion` cargado |

### `GET /v1/model`
Nombre, versión, alias, clases y métricas de evaluación del modelo cargado.

### `POST /v1/reload`
Fuerza la recarga del `champion` (la API también lo chequea cada 30 s).

### `GET /health`
`200 {"status": "ok", "model_loaded": true}` · `503` si el modelo no está cargado.

## 3. gRPC (Sesión 3)

Mismos campos: la imagen viaja como `bytes` (sin base64). `Detect` (unary) y
`DetectStream` (server-streaming sobre un lote de frames). El `.proto` está en
`proto/detector.proto`; `scripts/gen_proto.sh` genera el código. Puerto `50051`.
Errores: 413 → `RESOURCE_EXHAUSTED`, 415/422 → `INVALID_ARGUMENT`, 503 → `UNAVAILABLE`.

Además de `Detect` y `DetectStream`, el servicio expone `GetModel` (modelo cargado: versión, alias,
etiquetas, umbral calibrado y métricas; `loaded=false` si no hay champion) y `Reload` (busca el champion
en MLflow; `UNAVAILABLE` si no puede). Los usa el gateway REST para `/health`, `/v1/model` y `/v1/reload`.

**Topología.** El servicio de inferencia (`inference`) es el único que carga el modelo y persiste en
`predictions/`. El REST es un gateway sin modelo que valida tipo, tamaño y umbral, reenvía los bytes por
gRPC y traduce los errores (`INVALID_ARGUMENT` → 422, `RESOURCE_EXHAUSTED` → 413, `UNAVAILABLE` → 503,
`DEADLINE_EXCEEDED` → 504). `inference` se puede escalar con réplicas (`--scale inference=N`); el
gateway las reparte por DNS con `round_robin`. `latency_ms` mide solo la inferencia, no el salto del gateway.

## 4. Data Lake (MinIO)

Bucket `data`, organizado por zonas:

```
s3://data/
  raw/images/<dataset>/<split>/…            imágenes originales (train/val/test)
  raw/stream/date=YYYY-MM-DD/…              frames que llegan por streaming
  curated/eval/<dataset>/annotations.json   set de validación revisado (COCO, copia de annotations/reviewed/val.json)
  models/billboard-detector/v<N>/…          pesos + metadata versionados
  predictions/date=YYYY-MM-DD/<request_id>.json   salida de cada inferencia (formato §2 + `image_stats`)
  curated/reference/billboard-detector/v<N>.json  features del set de evaluación por versión (referencia de drift)
  monitoring/drift/<timestamp>.json               reportes de `monitoreo_drift`
```

Bucket `mlflow`: artefactos de MLflow (lo gestiona MLflow, no se escribe a mano).

## 5. DAGs (Airflow)

| DAG | Qué hace |
|---|---|
| `registrar_modelo` | descarga pesos → `models/…/v<N>/` → evalúa sobre `curated/eval` (mAP) → registra en MLflow → alias `champion` si supera el umbral y al campeón actual |
| `monitoreo_drift` | lee `predictions/` recientes del champion → compara contra la referencia con KS de dos muestras (`n_detections`, `score_mean`, `brightness`, `contrast`; α=0.01) → reporte en `monitoring/drift/`; la tarea falla si hay drift. Con menos de 20 frames no emite veredicto |

Cada predicción persistida agrega `image_stats` (`brightness` y `contrast` de la luminancia, 0–1); REST y gRPC no lo devuelven. `registrar_modelo` genera la referencia infiriendo `val` con el umbral calibrado.
