# Seguridad, operación y gobernanza (Sesión 7, SAIF)

Riesgos del sistema ordenados por las capas de SAIF (datos, modelo, infraestructura, aplicación),
el control implementado y lo que queda abierto. "Verificado" significa que hay un test o una
prueba contra el stack que lo demuestra.

## Superficie expuesta

| Servicio | Dónde escucha | Autenticación |
|---|---|---|
| Gateway REST + GraphQL | `${GATEWAY_BIND}:8800` (0.0.0.0 por defecto: es el único servicio pensado para la red) | API key (`X-API-Key`), admin para `/v1/reload` |
| Inference (gRPC) | red interna `backend`; `127.0.0.1:50051` para scripts locales | token en metadata (`INFERENCE_TOKEN`) |
| Airflow, MLflow, MinIO (API y consola), Postgres, Redpanda, Label Studio | solo `127.0.0.1` | usuario y contraseña de `.env` (MLflow: sin usuarios, ver residuales) |

Todos los secretos están en `.env`, que **no se versiona** y genera `scripts/generar_env.sh`
con valores aleatorios (permisos 600). `docker-compose.yaml` no tiene contraseñas por defecto:
si falta una variable, `docker compose` se niega a arrancar.

## Riesgos y controles

| Capa | Riesgo | Control | Estado |
|---|---|---|---|
| Aplicación | Uso del servicio por cualquiera en la red | API key por cliente, comparación en tiempo constante; `/v1/reload` solo con clave admin | ✅ verificado (`test_api.py`: 401/403) |
| Aplicación | Abuso / denegación de servicio por volumen | Rate limit por clave (ventana deslizante); límite de tamaño por `Content-Length` antes de leer el cuerpo (411/413) | ✅ verificado (429, 411, 413) |
| Aplicación | Imagen maliciosa: bomba de descompresión, formato inesperado | Formato real JPEG/PNG (no el tipo declarado); dimensiones validadas antes de decodificar; `DecompressionBombError` → 422 | ✅ verificado (GIF disfrazado → 415, PNG 20000² → 422 sin decodificar) |
| Aplicación | Filtrado de información interna en errores | Errores no previstos → 502/`INTERNAL` con mensaje genérico; el detalle va al log | ✅ verificado |
| Aplicación | GraphQL: consultas abusivas o descubrimiento del esquema | Sin introspección, límite de profundidad, `limite` ≤ 100, misma API key | ✅ verificado |
| Infraestructura | Llamadas directas al modelo saltando el gateway | `inference` solo en la red interna; token gRPC (interceptor) | ✅ verificado (`UNAUTHENTICATED`) |
| Infraestructura | Servicios de administración expuestos (Postgres, MinIO, MLflow, Airflow) | Publicados solo en `127.0.0.1`; MLflow con `--allowed-hosts` (anti DNS rebinding) en lugar de desactivar su middleware; Airflow sin `EXPOSE_CONFIG` | ✅ |
| Infraestructura | Secretos versionados o por defecto | `.env` fuera de git, generado; sin defaults en compose; Fernet/JWT/secret de Airflow propios de cada instalación | ✅ |
| Datos | Un servicio comprometido lee o borra todo el lake | Un usuario de MinIO por servicio con mínimo privilegio: `inference` lee `models/`, escribe `predictions/` y `raw/stream/`; `gateway` solo lee `predictions/`; `stream` lee referencias y escribe ventanas y alertas. Nadie más que Airflow puede escribir `models/` | ✅ (`dockerfiles/minio/init.sh`) |
| Datos | Datos de entrenamiento o evaluación alterados | Sets revisados versionados en git; en el lake con su sha256 en el nombre y registrado en MLflow | ✅ |
| Datos | Privacidad: caras y patentes en las imágenes guardadas | Solo se guarda una muestra (`PERSIST_IMAGES_RATIO`, 10 %) y solo para reetiquetar | ⚠️ parcial: sin desenfoque de caras y patentes (trabajo futuro) |
| Modelo | Pesos manipulados en el lake (cadena de suministro) | SHA-256 del zip del release obligatorio; SHA-256 del checkpoint registrado en MLflow y **verificado por inference al cargar** (si no coincide, no lo sirve) | ✅ |
| Modelo | Un modelo peor llega a producción | Gate sobre un set no usado en el entrenamiento ni la calibración, champion re-evaluado con el mismo set, margen mínimo; rollback con `rollback_modelo` | ✅ |
| Gobernanza | Saber qué modelo, datos y código produjeron cada predicción | Predicción → versión del modelo → `eval_run_id` → set de evaluación (artefacto + sha) + `code_sha256` + `git_commit` | ✅ |
| Gobernanza | Degradación silenciosa en producción | `monitoreo_drift` horario + ventanas del consumidor; alertas en `monitoring/alerts/` | ✅ |

## Riesgos residuales (conocidos y aceptados para este alcance)

- **MLflow sin usuarios.** Cualquier proceso en la red interna o en la máquina puede mover el alias
  `champion`. Mitigado: no se expone fuera de `127.0.0.1`, inference solo sirve pesos cuyo hash
  coincide con el registrado, y el alias anterior queda en `anterior`. Para producción: MLflow con
  `--app-name basic-auth` o detrás de un proxy con autenticación.
- **Sin TLS.** El tráfico interno (gRPC, S3, MLflow) va en claro dentro de la red de Docker y el
  gateway sirve HTTP. En un despliegue real: TLS en el borde (proxy) y mTLS entre servicios.
- **Rate limit en memoria.** Vale para una réplica de gateway; con varias haría falta un contador
  compartido (Redis/Valkey).
- **API keys estáticas.** Se rotan cambiando `.env` y reiniciando el gateway; no hay JWT con expiración.
- **Airflow y MLflow usan la cuenta root de MinIO** (administran el lake y los artefactos).
