# VallaVision — Detección de vallas publicitarias en producción

**TP integrador · Documento de alcance (BORRADOR)**

> Estado: borrador para discutir en el grupo. Los puntos marcados **[POR DEFINIR]** dependen de decisiones abiertas.
> Materia: MLOps II (CEIA · FIUBA). Hitos: diseño (Sesión 4) · checkpoint (Sesión 6) · defensa (Sesión 8) · repo una semana después.

**Equipo:** Christian Conchari (individual — confirmar con el docente, el reglamento pide 2–6)

## 1. Objetivo

Poner en producción un **detector de vallas publicitarias en imágenes de calle**, como servicio de predicción de la empresa ficticia *ML Models & Something More Inc.*

El foco es la **puesta en producción**, no el modelado. El modelo (un ViT de **detección**, en **PyTorch**) se desarrolla en otra materia y se **incorpora como artefacto ya entrenado**. Lo que se evalúa es todo lo que lo rodea: orquestación, ciclo de vida, almacenamiento, serving, monitoreo y documentación.

El **entrenamiento es opcional** (ver 2.3): solo se incorpora si el flujo de entrenamiento y los datos quedan claros y suficientes a tiempo. Si no, este TP no lo cubre y no afecta al resto del alcance.

**Nivel objetivo:** contenedores (8–10). Todo corre con Docker Compose y con instrucciones de instalación y ejecución.

## 2. En alcance

### 2.1 Núcleo (obligatorio)

| Componente | Qué se entrega |
|---|---|
| **MinIO (S3)** | Buckets para datos crudos (imágenes), artefactos del modelo y resultados de inferencia. |
| **MLflow** | Tracking y Model Registry. El modelo preentrenado se registra con versión, metadatos y métricas de evaluación, y se promueve entre etapas (`Staging` → `Production`). |
| **Airflow** | DAGs que orquestan el ciclo de vida del modelo (ver 4). |
| **FastAPI (REST)** | Endpoint de inferencia versionado (`/v1/detect`) y `/health`. Validación de entrada (tipo, tamaño, dimensiones) y respuesta con la versión del modelo. |
| **Docker Compose** | Un solo `docker compose up` levanta todo el sistema. |
| **Documentación** | README con arquitectura, instalación, ejecución y decisiones de diseño. |

### 2.2 Diferencial (capas adicionales comprometidas)

- **gRPC:** servicio de detección con imágenes como `bytes`, `Detect` unary y server-streaming para lotes de frames. Se compara latencia y tamaño de mensaje contra REST con base64.
- **Streaming:** un flujo de frames (video o carpeta de imágenes reproducida como una cámara) con inferencia evento a evento, métricas por ventana (throughput, latencia p95, detecciones por frame) y detección de **drift** (por ejemplo, sobre la confianza media y las estadísticas de la imagen).
- **GraphQL (mínimo, reutilizado):** consulta de metadatos del modelo activo y de detecciones. Se reutiliza lo ya hecho en el Mini-TP 2. **[POR CONFIRMAR si entra]**

### 2.3 Extensión opcional: entrenamiento

**Condición para incluirla** (decidir antes del checkpoint de la Sesión 6): un flujo de entrenamiento reproducible y un dataset claro, versionado y suficiente.

Si se incluye: un DAG `entrenar_modelo` que lee el dataset desde MinIO, entrena el ViT, registra parámetros, métricas (mAP) y artefacto en MLflow, y deja el modelo en `Staging` para pasar por `registrar_modelo`. Como el entrenamiento en CPU es lento, se ejecutaría fuera del orquestador (máquina con GPU) y solo se registraría el resultado.

Si no se incluye: se documenta como trabajo futuro y no hay cambios en el resto.

## 3. Fuera de alcance

- Optimizar el mAP del modelo o mejorar su arquitectura.
- Anotar nuevos datos o construir el dataset.
- Aprendizaje federado y capas de seguridad avanzadas (JWT, desenfoque de caras y patentes). Se mencionan como trabajo futuro en la defensa.
- Despliegue en una nube real, GPU en producción y autoescalado.
- Cámaras o hardware real.

## 4. Arquitectura propuesta

```
 frames / imágenes ──► MinIO (raw) ──┐
                                     ▼
 pesos del ViT ──► Airflow: registrar_modelo ──► MLflow Registry ──► MinIO (artefactos)
                                                       │
                                                       ▼ (modelo en Production)
                  ┌─────────────── servicio de inferencia ───────────────┐
 cliente REST ───►│ FastAPI  /v1/detect                                    │
 cliente gRPC ───►│ gRPC     Detect / DetectStream                         │──► MinIO (resultados)
 productor ──────►│ consumidor de streaming + métricas de ventana + drift  │
                  └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                     Airflow: monitoreo_drift ──► reporte / alerta
```

### DAGs de Airflow

1. **`registrar_modelo`**: toma los pesos del ViT, los sube a MinIO, corre una evaluación sobre un conjunto de validación pequeño, registra métricas y versión en MLflow y promueve a `Production` si supera un umbral.
2. **`monitoreo_drift`**: periódico. Lee las detecciones recientes, calcula estadísticas contra una referencia y emite un reporte o alerta.

## 5. Decisiones abiertas

| # | Decisión | Opciones | Impacto |
|---|---|---|---|
| 1 | ✅ Tarea del modelo | **Detección** (cajas + clase + confianza) | Contrato REST, `.proto` y métricas (mAP). |
| 2 | ✅ Formato del modelo | **PyTorch**. Exportar a ONNX o TorchScript queda como optimización opcional | Latencia y tamaño de imagen Docker. |
| 2b | Incluir entrenamiento | Sí o no (ver 2.3) | DAG extra y necesidad de datos versionados. |
| 3 | Salida del detector | Formato de cajas y clases, umbral de confianza | Contratos y esquema de resultados en MinIO. |
| 4 | Fuente del flujo | Video propio o carpeta de imágenes reproducida | Realismo del streaming. |
| 5 | Referencia de drift | Qué estadísticas y contra qué conjunto | Calidad del monitoreo. |
| 6 | GraphQL | Entra o no | Alcance del diferencial. |
| 7 | Convivencia con la otra materia | Confirmar con los docentes el uso compartido del modelo | Riesgo académico. |

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El ViT es lento en CPU | Medir y documentar la latencia; si hace falta, cuantizar o exportar a ONNX. |
| Imagen de Docker muy pesada (PyTorch) | Instalar la versión CPU de torch, construcción multi-etapa y una sola imagen base compartida entre REST y gRPC. |
| El entrenamiento consume el tiempo del TP | Es opcional y con condición de entrada explícita (2.3); el núcleo no depende de él. |
| El modelo de la otra materia no llega a tiempo | Usar un detector preentrenado público como sustituto con el mismo contrato, para no bloquear la infraestructura. |
| Airflow y MLflow consumen mucha memoria en local | Definir los recursos mínimos en el README y usar imágenes ligeras. |
| Alcance que crece | Este documento manda: todo lo nuevo pasa por revisión del grupo. |

## 7. Plan de hitos

| Hito | Fecha | Entregable |
|---|---|---|
| Hito #1 · diseño (Sesión 4) | [POR DEFINIR] | Este documento, diagrama de arquitectura y plan. |
| Núcleo funcionando | Antes de la Sesión 6 | Compose con MinIO, MLflow y Airflow; DAG `registrar_modelo`; REST `/v1/detect`. |
| Checkpoint (Sesión 6) | [POR DEFINIR] | Demo del núcleo más avance de gRPC. |
| Diferencial completo | Antes de la Sesión 8 | gRPC, streaming con drift y DAG `monitoreo_drift`. |
| Defensa (Sesión 8) | [POR DEFINIR] | Presentación con los slides. |
| Repo final | 1 semana después | Repo con README, instrucciones y slides; link por el aula virtual. |

## 8. Criterio de éxito

- Un `docker compose up` levanta el sistema completo y una persona ajena al grupo lo reproduce siguiendo el README.
- El modelo se registra, versiona y promueve mediante Airflow y MLflow, sin pasos manuales ocultos.
- La inferencia funciona por REST y por gRPC, y el streaming produce métricas y alertas de drift.
- La documentación explica la arquitectura, las decisiones y lo que quedó fuera de alcance.

## 9. Datos y métodos (para el formulario del hito)

### Datos

> Imágenes de calle con anotaciones de vallas publicitarias (caja delimitadora + clase), en formato **[POR DEFINIR: COCO / YOLO]**, divididas en entrenamiento, validación y prueba. Origen: **[POR DEFINIR: dataset propio, público o de la otra materia]**. Se almacenan en un bucket de MinIO (S3) con versionado. Para el flujo en tiempo real se reproduce un video o una carpeta de imágenes como si fuera una cámara. El conjunto de validación se usa para evaluar el modelo antes de promoverlo en MLflow. La salida de cada inferencia (cajas, clases, confianza, versión del modelo) también se guarda en MinIO.

### Método a usar

| Método | Decisión | Cómo se usa |
|---|---|---|
| **gRPC** | ✅ Sí | Frames como `bytes`, `Detect` unary y server-streaming para lotes. Se compara la latencia contra REST con base64. |
| **Streaming** | ✅ Sí | Inferencia evento a evento, métricas por ventana (throughput, p95, detecciones por frame) y drift. |
| **Cloud** | ✅ Sí (local) | MinIO como Data Lake compatible con S3, simulado con Docker Compose. |
| **GraphQL** | ⚠️ Opcional | Metadatos del modelo activo y consulta de detecciones. Reutiliza el Mini-TP 2. |
| **FL** | ❌ No (trabajo futuro) | Se plantea en la defensa como evolución: clientes que entrenan con sus propias imágenes sin centralizarlas. |

Versión corta:
- **Datos:** imágenes de calle con cajas de vallas en formato **[POR DEFINIR]**, almacenadas en MinIO.
- **Método:** gRPC, Streaming y Cloud (MinIO/S3), con GraphQL mínimo opcional. FL queda como trabajo futuro.
