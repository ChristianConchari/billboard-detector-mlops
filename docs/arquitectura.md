# Arquitectura y decisiones

Resumen para la defensa. El contrato está en [contrato.md](contrato.md), los controles de seguridad en
[seguridad.md](seguridad.md) y el diseño original (Hito #1) en [alcance.md](alcance.md).

## Flujo

```mermaid
flowchart LR
    GH[GitHub release<br/>model-v1] -->|gh:// + SHA-256| REG
    DS[(dataset local<br/>data/ + annotations/reviewed/)] --> REG
    subgraph Airflow
        REG[registrar_modelo]
        RB[rollback_modelo]
        MON[monitoreo_drift<br/>@hourly]
    end
    REG -->|pesos v&lt;N&gt;, sets con sha, referencia| S3[(MinIO<br/>bucket data)]
    REG -->|modelo, evaluaciones, linaje, alias| ML[MLflow]
    RB -->|champion ↔ anterior| ML
    ML -->|alias champion + sha256| INF
    S3 -->|pesos verificados| INF
    C1[cliente REST] -->|X-API-Key| GW[Gateway :8800<br/>sin modelo]
    GW -->|gRPC + token| INF[inference :50051<br/>modelo + torch<br/>réplicas con --scale]
    CAM[camara_productor.py] -->|topic frames| RP[(Redpanda)]
    RP --> SC[stream-consumer<br/>métricas por ventana de event time]
    SC -->|gRPC + token| INF
    SC -->|curated/stream_windows/, alertas| S3
    INF -->|predictions/| S3
    S3 -->|predictions/ + referencia| MON
    MON -->|reporte y alerta| S3
```

## Decisiones

| Decisión | Elegido | Por qué |
|---|---|---|
| Promoción de modelos | **Aliases** de MLflow 3 (`champion`, `challenger`, `anterior`) | Las *stages* están deprecadas; `anterior` hace el rollback de un paso |
| Set del gate | `test` revisado (95 imágenes, 7 videos) | No intervino en el entrenamiento ni en la calibración del umbral; el `val` revisado sí (por eso daba 0,817, optimista) |
| Comparación contra el champion | Re-evaluarlo sobre el mismo set (o reutilizar una evaluación con el mismo sha256 del set) y exigir una mejora ≥ 0,01 | Comparar métricas de sets distintos no dice nada; un empate dentro del ruido no justifica cambiar el modelo |
| Incertidumbre | IC 95 % del mAP por bootstrap de imágenes, publicado con la métrica | Con 95 imágenes el IC mide ±0,05: hay que mostrarlo |
| Linaje | Cada evaluación es un run (`evaluaciones`) con el set como artefacto, su sha256, el hash del código y el commit; la versión del modelo apunta a su última evaluación | Reproducir "qué datos y qué código dieron esta métrica" sin depender de paths mutables |
| Integridad de pesos | SHA-256 del zip del release obligatorio; SHA-256 del checkpoint en MLflow, verificado por inference al cargar | Un peso alterado en el lake no llega a servirse |
| Carga en el servicio | `BillboardDetector` desde MinIO, no `pyfunc.load_model` | Controlar umbral, hilos y latencia por request; el pyfunc queda registrado para consumo externo |
| Código compartido | Paquete `vallavision/` montado en Airflow y copiado a las imágenes | Lo que se evalúa es exactamente lo que se sirve |
| Separación gateway / inferencia | REST en un gateway sin torch ni acceso al lake; el modelo solo en `inference` (gRPC unario, interno) | Una copia del modelo por réplica, borde liviano, escalado independiente y un lugar para autenticación y límites |
| Validación | Baratos en el gateway; formato real, dimensiones (antes de decodificar) y EXIF en inference | Los clientes gRPC directos quedan igual de protegidos |
| Umbral por defecto | El calibrado (`calibration.json`, 0,15) | Viaja con el checkpoint |
| Streaming | Redpanda + consumidor con ventanas tumbling por event time, watermark y cierre por inactividad | Las cámaras producen más rápido de lo que inference procesa (~2,4–4 fps): hace falta un buffer durable. El broker ordena por cámara (partición) y el event time hace las ventanas reproducibles aunque lleguen frames desordenados |
| Drift | En un solo lugar (`monitoreo_drift`, sobre todo el tráfico): chi² para el conteo, KS para las continuas, Holm sobre los 4 tests; solo predicciones con el umbral calibrado | Una sola política de alertas; el conteo es discreto; sin corrección las falsas alarmas se multiplican; otro umbral cambia las features |
| Monitoreo → acción | Alerta en `monitoring/alerts/` y corrida en rojo; decide una persona | El reentrenamiento vive en el repo del modelo, fuera de este sistema |
| Persistencia de predicciones | Best-effort; no se persiste si el cliente ya se fue | La inferencia no depende del monitoreo y el monitoreo no ve requests abandonadas |

## Resultados medidos

Instalación limpia (`down -v` y `.env` nuevo) en un i7-12700F (20 hilos), CPU, 2026-10-15.

| Qué | Resultado |
|---|---|
| `registrar_modelo` de punta a punta | 7 tareas OK en **3,7 min**; promovió solo a `champion` |
| Gate (`test` revisado, 95 imágenes) | **mAP 0,647 (IC 95 % 0,611–0,705) · AP50 0,930**; el release reporta 0,649 / 0,930 en el mismo set |
| Linaje en MLflow | run de evaluación con `eval_set/test-11bd28e3bec4.json`, `eval_sha256`, `code_sha256`, `git_commit` = HEAD; versión con `weights_dir_sha256` y `weights_source_sha256` |
| Pesos alterados en MinIO | inference no los carga (`IntegrityError`, hash distinto del de MLflow) → `/health` 503; al restaurarlos vuelve solo |
| Validación (ataques de la evaluación) | GIF declarado `image/jpeg` → **415**; PNG 20000² → **422** sin decodificar; BMP por gRPC → `INVALID_ARGUMENT` (415 en el trailer) |
| Superficie | sin clave → 401, clave inválida → 403, reload con clave de usuario → 403; gRPC sin token → `UNAUTHENTICATED`; un usuario de servicio de MinIO no puede escribir `models/` ni leer `mlflow`; MLflow rechaza `Host` ajeno (403) |
| Streaming con broker | 3 cámaras, ventanas de 15 s por event time: p95 ≈ 440 ms, ~1,7 vallas por frame |
| `monitoreo_drift`, controles negativo y positivo | con videos fuera de la referencia, sin drift; con frames oscurecidos (319 frames con umbral calibrado): drift en brillo (p = 2e-20) y contraste (p = 5e-13), no en conteo ni score; alerta y Parquet en `curated/predictions/` |
| Falsas alarmas bajo H0 (test unitario) | ≤ 6 de 200 ventanas con Holm a α = 0,01 |
| Tamaño de request (imagen de 285 KB) | gRPC 285 103 B · multipart 285 099 B · JSON+base64 380 144 B (+33 %) |
| Separación gateway / inferencia | imagen de borde 2,3 GB → **~245 MB**; RAM 1448 → **647 MiB** (una copia del modelo) |
| Escalado a igual CPU (8 hilos, REST, 40 requests) | 1 réplica × 8 hilos: **3,3 req/s, p50 291 ms** con 1 cliente; 2 réplicas × 4 hilos: **4,0 req/s** con 2–4 clientes (**+20 %**) pero p50 441 ms con 1 cliente. Contra 1 × 4 hilos (2,4 req/s) el ×1,7 de antes comparaba el doble de CPU. Requiere `max_connection_age` para que el gateway re-resuelva el DNS |
| Tests | 72, más el CI con `docker compose config` y la compilación de los DAGs |

## Qué se dejó afuera y por qué

Criterio: cada pieza tiene que tener un consumidor o una razón operativa. Estas se implementaron o se
evaluaron y se sacaron:

| Pieza | Por qué no está |
|---|---|
| GraphQL | Ningún cliente compone consultas sobre modelo, registro y predicciones. Obligaba al gateway a leer MLflow y MinIO (más dependencias y superficie); dos endpoints REST harían lo mismo si hiciera falta |
| RPCs de streaming (`DetectStream`, `DetectLive`) | Tres caminos de streaming para una sola necesidad: el broker ya da buffer, orden por cámara y replay. gRPC queda para llamadas internas unarias |
| Drift en el consumidor de streaming | Duplicaba el cálculo y la política de alertas de `monitoreo_drift`, que además ve el tráfico REST |
| Guardar una muestra de imágenes + manifiesto de reetiquetado | Sin un pipeline de etiquetado que los consuma, guardar fotos de la calle (caras, patentes) es costo de privacidad sin beneficio |
| Label Studio | Herramienta de curación del dataset de origen; el trabajo terminó (se usan las cajas revisadas del repo del modelo) y no es parte del sistema en producción |
| Aprendizaje federado | Un solo detector entrenado de forma central: sería una simulación sin caso real |

## Limitaciones conocidas

- **Set del gate chico.** 95 imágenes de 7 videos: el IC 95 % mide ±0,05. Sirve para frenar un modelo roto o
  claramente peor, no para distinguir diferencias de 0,01; por eso el margen mínimo y el IC publicado. Si el gate se usa
  muchas veces para elegir entre candidatos, el `test` se va gastando: haría falta un holdout nuevo periódico.
- **Drift.** La referencia es el `val` revisado (95 frames de 7 videos), no tráfico real; con una sola cámara real lo
  esperable es drift casi siempre hasta construir una referencia por cámara. Frames consecutivos están autocorrelacionados
  (los tests suponen independencia): `thin` ayuda pero no lo resuelve. El drift no mide la calidad del modelo: para eso
  hace falta ground truth, que este sistema no recolecta.
- **Latencia y escalado.** ~290–440 ms por imagen en CPU (8 o 4 hilos) frente a 19,5 ms en GPU. A igual CPU, una
  réplica con más hilos da menor latencia y dos réplicas dan ~20 % más throughput con varios clientes: replicar conviene
  para throughput y disponibilidad, no para latencia. ONNX Runtime o int8 quedan como trabajo futuro.
- **Streaming.** El productor simula cámaras con imágenes del dataset; semántica at-least-once (un frame puede
  procesarse dos veces si el consumidor se cae antes del commit).
- **Seguridad.** Residuales en [seguridad.md](seguridad.md): MLflow sin usuarios, sin TLS, rate limit en memoria.

## Trabajo futuro

- Referencias de drift por cámara.
- Un ciclo de etiquetado (muestreo de imágenes con desenfoque de caras y patentes) si se decide reentrenar con tráfico real.
- ONNX o cuantización para bajar la latencia en CPU; TLS/mTLS y MLflow con autenticación.
