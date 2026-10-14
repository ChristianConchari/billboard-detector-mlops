# Arquitectura y decisiones

Resumen para la defensa. El contrato de datos está en [contrato.md](contrato.md) y el alcance
original en [alcance.md](alcance.md).

## Flujo

```mermaid
flowchart LR
    GH[GitHub release<br/>model-v1] -->|gh:// + SHA-256| REG
    DS[(dataset local<br/>data/)] --> REG
    subgraph Airflow
        REG[registrar_modelo]
        MON[monitoreo_drift<br/>@hourly]
    end
    REG -->|pesos v&lt;N&gt;, dataset, referencia| S3[(MinIO<br/>bucket data)]
    REG -->|modelo + métricas + alias| ML[MLflow Registry]
    ML -->|alias champion| SRV
    S3 -->|pesos| SRV
    subgraph SRV[Servicio de inferencia]
        REST[Gateway REST<br/>:8800 · sin modelo] -->|gRPC| GRPC[inference<br/>:50051 · modelo + torch<br/>réplicas con --scale]
    end
    C1[cliente REST] --> REST
    C2[stream_client.py<br/>cámara simulada] --> GRPC
    GRPC -->|predictions/| S3
    S3 -->|predictions/ + referencia| MON
    MON -->|monitoring/drift/| S3
```

## Decisiones

| Decisión | Elegido | Por qué |
|---|---|---|
| Promoción de modelos | **Aliases** de MLflow 3 (`champion`/`challenger`) | Las *stages* están deprecadas en MLflow 3 |
| Carga en el servicio | `BillboardDetector` directo desde MinIO, no `pyfunc.load_model` | Hace falta controlar latencia, umbral y tamaño por request |
| Código compartido | Paquete `vallavision/` en la raíz, montado en Airflow y copiado a la imagen de la API | Lo que se evalúa es exactamente lo que se sirve |
| Separación gateway / inferencia | REST en un gateway sin torch; el modelo vive solo en `inference` (gRPC) | Una sola copia del modelo por réplica, imagen de borde liviana, escalado independiente y lugar para la seguridad de la Sesión 7 |
| Validación | Chequeos baratos en el gateway; decodificado y dimensiones en `inference` (`validation.py` compartido) | El gateway no decodifica imágenes y los clientes gRPC directos quedan igual de protegidos |
| Umbral por defecto | El calibrado sobre validación (`calibration.json`, 0.15), no un valor fijo | Viaja con el checkpoint y es el que maximiza F1 |
| Fuente de pesos | `gh://owner/repo@tag/asset.zip` con verificación SHA-256 | Reproducible sin pasos manuales ni credenciales |
| Idempotencia | Mismos pesos (sha256) → misma `v<N>` y misma versión de MLflow | Re-ejecutar el DAG no duplica nada |
| Drift | Test KS de dos muestras por feature (α = 0,01) contra `val` | Sin supuestos de distribución y sin entrenar nada |
| Persistencia de predicciones | Best-effort: si MinIO falla no se rompe la request | La inferencia no depende del monitoreo |

## Resultados medidos

| Qué | Resultado |
|---|---|
| `registrar_modelo` de punta a punta | 6 tareas OK en ~17 min en CPU (evaluar y referencia corren en paralelo) |
| REST `/v1/detect` con el RT-DETR | Contrato correcto; 415, 422 y umbral inválido verificados |
| gRPC `Detect` y `DetectStream` | Funcionan; `DetectStream` devuelve un frame por respuesta, en orden |
| Tamaño de request (imagen de 209 KB) | gRPC 209 284 B · multipart 209 280 B · JSON+base64 279 052 B (+33 %) |
| Streaming normal | Sin drift en 2 ventanas de 20 frames |
| Streaming con frames oscurecidos | Drift en `brightness` y `contrast` en ambas ventanas |
| Separación gateway / inferencia | Imagen de borde 2.3 GB → **245 MB**; RAM 1448 → **647 MiB** (una sola copia del modelo); build del gateway 31 s; resultado idéntico al anterior |
| Escalado con `--scale inference=2` | **Sin demostrar.** Primer intento: el gateway usaba solo la réplica 1; se corrigió con `max_connection_age` y falta re-medir |
| Tests | 40 (gateway con stub gRPC, servicio, servidor gRPC real, drift) |
| Latencia de inferencia en CPU | ~1,5–2 s por imagen (8 núcleos, solo AVX2); el forward domina, el preprocesamiento son 11 ms |

## Limitaciones conocidas

- **mAP de validación.** El DAG mide mAP 0,066 sobre `val` de este dataset, contra 0,817 (val) y 0,649 (test)
  del release. No es un fallo de carga: la inferencia coincide con la del repo del modelo. El split por
  video difiere del de entrenamiento y 102 de las 169 cajas de `val` son pre-etiquetas `pending` sin revisar.
  Hasta evaluar con anotaciones revisadas, `min_map = 0.5` deja al modelo como `challenger`.
- **Latencia.** Unos 2 s por imagen en CPU frente a 19,5 ms en GPU. Opciones: ONNX Runtime, cuantización
  int8 o GPU. No se midieron en una máquina sin carga.
- **Streaming simulado.** Una carpeta de imágenes reproducida a N fps por gRPC unario, no un broker de
  mensajes. A 2 s por frame el throughput efectivo es ~0,5 fps.
- **Drift.** Con ventanas chicas y 4 features hay riesgo de falsas alarmas (comparaciones múltiples sin
  corrección). La referencia es `val`, que no es tráfico real.

## Trabajo futuro

- Aprendizaje federado: clientes que ajustan el detector con sus propias imágenes sin centralizarlas.
- Seguridad: autenticación (JWT) y desenfoque de caras y patentes antes de persistir.
- ONNX o cuantización para bajar la latencia en CPU.
- GraphQL para consultar el modelo activo y las detecciones (opcional en el alcance).
