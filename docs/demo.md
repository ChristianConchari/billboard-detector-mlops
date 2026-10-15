# Guion de la demo grabada

`scripts/demo.sh` corre los segmentos en orden: muestra cada comando, espera **Enter** y recién ahí
lo ejecuta, así el ritmo lo marca quien graba. Los secretos se leen de `.env` y en pantalla aparecen
como `$GATEWAY_API_KEYS`. Se puede grabar por partes (`scripts/demo.sh 6`) y unir después.

## Antes de grabar

- [ ] Stack arriba y con champion (`curl localhost:8800/health` → `model_loaded: true`). No construir en cámara.
- [ ] `.env` propio para la grabación (`mv .env .env.real && scripts/generar_env.sh` y `down -v` + `up` + `registrar_modelo`), o al menos no abrirlo nunca en pantalla.
- [ ] Cerrar navegador pesado y otros proyectos: OBS más el modelo compiten por la CPU y la latencia sube.
- [ ] Terminal con fuente grande (≥ 16 pt), prompt corto, 120 columnas. Para la terminal, `asciinema rec` da un video limpio; OBS para el navegador.
- [ ] Pestañas abiertas: Airflow (grafo de `registrar_modelo`), MLflow (experimento `evaluaciones`), MinIO (bucket `data`).
- [ ] Ensayo sin pausas: `DEMO_SIN_PAUSAS=1 DEMO_NO_ABRIR=1 scripts/demo.sh` (≈ 9 min, deja el sistema como estaba).

## Segmentos

| # | Qué se ve | Duración | Qué decir |
|---|---|---|---|
| 1 | `docker compose ps`: los servicios healthy y sus puertos | 30 s | "Todo el sistema es un `docker compose up`. Solo el gateway escucha fuera de la máquina." |
| 2 | `registrar_modelo` disparado; las tareas terminan una a una (cortar la espera); resumen de MLflow: alias, mAP 0.647 con IC, motivo de la promoción, sha de los pesos y commit | 1,5 min | "Ningún modelo llega a producción sin pasar un gate sobre datos que no vio, comparado con el champion sobre el mismo set, y con linaje completo." Mostrar en la UI de MLflow el run de evaluación con el set como artefacto. |
| 3 | La foto de la calle con las 5 vallas detectadas y el JSON | 45 s | "El gateway no tiene el modelo: valida, autentica y reenvía por gRPC. La respuesta trae la versión del modelo." |
| 4 | 401 sin clave, 403 con clave inválida, 415 a un GIF disfrazado, 403 al recargar sin clave admin, `UNAUTHENTICATED` por gRPC | 45 s | "Los controles de la Sesión 7, cada uno con su test." |
| 5 | Ventanas por cámara: frames, p95, vallas por frame; primero tráfico normal, después de noche (oscurecido) | 1,5 min | "Redpanda absorbe la diferencia entre lo que producen las cámaras y lo que infiere el modelo, y ordena por cámara; las ventanas son por el instante de captura, no de llegada." |
| 6 | `monitoreo_drift` sobre ese tráfico: falla a propósito; reporte con los p-valores (brillo y contraste) y la alerta | 1 min | "El drift se calcula en un solo lugar, sobre todo el tráfico, y alerta a una persona." |
| 7 | (opcional) Pesos alterados en MinIO → inference se niega a servirlos → restaurados | 45 s | "La cadena de suministro del modelo: el hash registrado en MLflow manda." |

Duración total: 6–7 min. El escalado, la latencia y REST vs gRPC van mejor en una diapositiva con las
tablas de [arquitectura.md](arquitectura.md) que en vivo, igual que "Qué se dejó afuera y por qué".

## Si algo sale mal

- Segmento 7 interrumpido con Ctrl+C: restaura los pesos solo; si no, `scripts/demo.sh 7` de nuevo los deja bien.
- Segmento 5 sin ventanas: `docker logs billboard-detector-mlops-stream-consumer-1` (el consumidor tiene que estar corriendo).
- Las demos escriben en `predictions/`: el `monitoreo_drift` de la hora siguiente va a ver ese tráfico (y el oscurecido da drift). Para que el segmento 6 muestre drift, correr antes el 5.
