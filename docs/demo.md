# Guion de la demo grabada

`scripts/demo.sh` corre los segmentos en orden: muestra cada comando, espera **Enter** y recién ahí
lo ejecuta, así el ritmo lo marca quien graba. Se puede grabar por partes (`scripts/demo.sh 6`) y unir después.

**Lo que se ve mejor en una interfaz va en el navegador; lo demás, en la terminal.** Los DAGs se
disparan desde Airflow (segmentos 2 y 6: el script espera la corrida, muestra las tareas y resume en la
terminal) y la inferencia se prueba en `/docs` del gateway (segmento 3). Para ensayar sin la interfaz:
`DEMO_CLI=1` dispara los DAGs por comando.

Los secretos se leen de `.env` y nunca se imprimen: en pantalla aparecen como `$GATEWAY_API_KEYS`. Para
`/docs`, el script copia la API key al portapapeles; se pega en *Authorize* (el campo la muestra un
momento: por eso conviene un `.env` propio para la grabación).

## Antes de grabar

- [ ] Stack arriba y con champion (`curl localhost:8800/health` → `model_loaded: true`). No construir en cámara.
- [ ] `.env` propio para la grabación (`mv .env .env.real && scripts/generar_env.sh` y `down -v` + `up` + `registrar_modelo`), o al menos no abrirlo nunca en pantalla.
- [ ] Cerrar navegador pesado y otros proyectos: OBS más el modelo compiten por la CPU y la latencia sube.
- [ ] Terminal con fuente grande (≥ 16 pt), prompt corto, 120 columnas. Para la terminal, `asciinema rec` da un video limpio; OBS para el navegador.
- [ ] Pestañas abiertas: Airflow (grafo de `registrar_modelo`), MLflow (experimento `evaluaciones`), MinIO (bucket `data`).
- [ ] Ensayo sin pausas: `DEMO_SIN_PAUSAS=1 DEMO_NO_ABRIR=1 scripts/demo.sh` (≈ 9 min, deja el sistema como estaba).

## Pantalla

Terminal a la izquierda (fuente ≥ 16 pt) y navegador a la derecha, con estas pestañas abiertas y la
sesión ya iniciada: Airflow (`http://localhost:8080`), MLflow (`http://localhost:5001`), MinIO
(`http://localhost:9001`) y `http://localhost:8800/docs`.

## Segmentos

| # | Dónde | Qué se hace y qué se ve | Duración | Qué decir |
|---|---|---|---|---|
| 1 | terminal | `docker compose ps`: los servicios healthy y sus puertos | 30 s | "Todo el sistema es un `docker compose up`. Solo el gateway escucha fuera de la máquina." |
| 2 | **Airflow** → **MLflow** | En Airflow: Dags → `registrar_modelo` → Trigger → vista Graph; las 7 tareas se ponen verdes (cortar la espera); log de `promover`. La terminal resume alias, mAP 0.647 con IC, motivo, sha y commit. En MLflow: experimento `evaluaciones` → último run → artefacto `eval_set/…` y tag `git_commit`; Models → `billboard-detector` con el alias `champion` | 1,5 min | "Ningún modelo llega a producción sin pasar un gate sobre datos que no vio, comparado con el champion sobre el mismo set, y con linaje completo." |
| 3 | **`/docs`** → foto | En `/docs`: Authorize (pegar la clave) → `POST /v1/detect` → Try it out → imagen → Execute: el JSON con las cajas y la versión. Después la terminal abre la foto con las 5 vallas encuadradas | 1 min | "El gateway no tiene el modelo: autentica, valida y reenvía por gRPC. La respuesta trae la versión que la produjo." |
| 4 | terminal | 401 sin clave, 403 con clave inválida, 415 a un GIF disfrazado, 403 al recargar sin clave admin, `UNAUTHENTICATED` por gRPC | 45 s | "Los controles de la Sesión 7, cada uno con su test." |
| 5 | terminal | Ventanas por cámara (frames, p95, vallas por frame), primero normal y después de noche; al final las particiones del topic y el retraso (*lag*) del grupo de consumidores | 2 min | "Redpanda absorbe la diferencia entre lo que producen las cámaras y lo que infiere el modelo. La cámara es la clave: sus frames quedan en orden. Las ventanas son por el instante de captura." |
| 6 | **Airflow** → terminal → **MinIO** | En Airflow: disparar `monitoreo_drift`; `alertar` queda en rojo a propósito. La terminal muestra los p-valores: drift en brillo y contraste, no en conteo ni score. En MinIO: `monitoring/alerts/` | 1 min | "El drift se calcula en un solo lugar, sobre todo el tráfico, y alerta a una persona. Detecta cambios; no mide calidad." |
| 7 | terminal | (opcional) Pesos alterados en MinIO → inference se niega a servirlos → restaurados | 45 s | "El hash registrado en MLflow manda." |

Versión corta para 15 minutos de presentación (≈ 4 min): segmentos **2, 3, 5 y 6**
(`scripts/demo.sh 2 3 5 6`). El 6 analiza el tráfico del 5: grabarlos en ese orden y seguidos.

El escalado, la latencia y REST vs gRPC van mejor en una diapositiva con las tablas de
[arquitectura.md](arquitectura.md) que en vivo, igual que "Qué se dejó afuera y por qué".

## Si algo sale mal

- Segmento 7 interrumpido con Ctrl+C: restaura los pesos solo; si no, `scripts/demo.sh 7` de nuevo los deja bien.
- Segmento 5 sin ventanas: `docker logs billboard-detector-mlops-stream-consumer-1` (el consumidor tiene que estar corriendo).
- Las demos escriben en `predictions/`: el `monitoreo_drift` de la hora siguiente va a ver ese tráfico (y el oscurecido da drift). Para que el segmento 6 muestre drift, correr antes el 5.
