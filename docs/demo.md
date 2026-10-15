# Guion de la demo grabada

La demo combina dos cosas, cada una donde se ve mejor:

- **Navegador** para lo que ya tiene interfaz: Airflow (disparar los DAGs y ver el grafo), MLflow (evaluaciones,
  linaje, alias) y MinIO (el Data Lake, las alertas).
- **Notebook** (`notebooks/billboard_detector_demo.ipynb`) para el servicio por dentro, con el código a la vista:
  inferencias con las cajas dibujadas, REST vs gRPC, los ataques de seguridad, el streaming de los vehículos y el drift
  con gráficos. Cada celda es corta y se entiende sola (código en inglés, explicación en español).

El relato es el del caso de uso: el sistema como el microservicio de detección de una **plataforma de
etiquetado geolocalizado de imágenes** (vehículos de relevamiento con cámara; cada frame con su posición GPS, visible
en las fotos). Sin detalles de la plataforma: solo su papel como consumidora del servicio.

```bash
python3 -m venv .venv-demo && .venv-demo/bin/pip install -r notebooks/requirements.txt   # una vez
.venv-demo/bin/jupyter lab notebooks/billboard_detector_demo.ipynb
```

El notebook lee los secretos de `.env` y no los imprime.

## Preparar la demo: el modelo inicial

La demo arranca con un modelo peor en producción, para que en la §7 se vea al gate promover uno mejor. El modelo inicial es
el release [`model-v0`](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v0) del proyecto de
CV3: el mismo RT-DETR entrenado con solo 66 imágenes (mAP 0.545 en el gate, IC 0.508–0.611). Se registra una vez, antes de
grabar, con el stack recién instalado:

```bash
docker exec airflow-apiserver airflow dags unpause register_model
docker exec airflow-apiserver airflow dags trigger register_model --conf '{
  "weights_source": "gh://ChristianConchari/vit-billboard-detector@model-v0/rtdetr-billboard.zip",
  "weights_sha256": "3af7213ce1bfbdd25eb64207ec05af6892e46ab69c0963bd94e08fc1a889b61c"}'
```

Queda como v1 `champion`. En la §7, `register_model` con los valores por defecto trae `model-v1` (441 imágenes, mAP 0.647):
el gate lo promueve a v2 `champion` y v1 pasa a `previous`. MLflow numera las versiones por orden de registro: la v2 del
registro es el release `model-v1`.

## Antes de grabar

- [ ] Stack arriba y con `model-v0` como único modelo registrado (v1 `champion`; `curl localhost:8800/health` → `model_loaded: true`). No construir en cámara.
- [ ] `.env` propio para la grabación (`mv .env .env.real && scripts/generate_env.sh` y `down -v` + `up` + el registro de `model-v0`), o al menos no abrirlo nunca en pantalla.
- [ ] Sin benchmarks en la hora previa: contaminan el drift que ve `drift_monitoring`.
- [ ] Cerrar navegador pesado y otros proyectos: el grabador y el modelo compiten por la CPU y la latencia sube.
- [ ] JupyterLab con zoom ≥ 125 % y el notebook sin salidas (Kernel → Restart Kernel and Clear Outputs).
- [ ] Pestañas abiertas y con la sesión iniciada: Airflow, MLflow, MinIO.
- [ ] Ensayo: el notebook entero promueve `model-v1`; después del ensayo hay que reinstalar y registrar `model-v0` de nuevo.

## Orden de la grabación

| # | Dónde | Qué se ve | Qué decir |
|---|---|---|---|
| 1 | Notebook, intro + celda 1 | el caso de uso (tabla de piezas y su papel en la plataforma) y el modelo servido: versión, mAP del gate con IC | "El sistema es el servicio de detección de una plataforma de etiquetado geolocalizado. Esto es lo que está sirviendo." |
| 2 | **Airflow → MLflow** | la corrida de `register_model` que registró `model-v0` (hecha antes de grabar; vista Graph) → log de `promote` → en MLflow el run de `evaluations` (métricas, `git_commit`, artefacto `eval_set`) y v1 con alias `champion`. **No dispararlo acá**: la §7 espera que el próximo registro sea la v2 | "Ningún modelo nuevo cambia las etiquetas de la plataforma sin pasar un gate sobre datos que no vio, con linaje completo." |
| 3 | Notebook §1 | la foto con las 5 vallas y su tabla; el mosaico de 4 recorridos con la posición GPS sobreimpresa | "La plataforma manda un frame; el servicio devuelve las cajas y la versión del modelo que las produjo." |
| 4 | Notebook §2 | REST vs gRPC: bytes y latencia; base64 +33 % | "Adentro, gRPC: bytes tal cual, contrato tipado. La plataforma solo ve el gateway." |
| 5 | Notebook §3 | la tabla de ataques, todo ✔ | "Los controles de seguridad, contra el sistema real." |
| 6 | Notebook §4–5 | los vehículos publican en Redpanda; ventanas por recorrido; particiones del topic; después de noche | "El broker absorbe la diferencia entre lo que producen los vehículos y lo que infiere el modelo; las ventanas son por instante de captura." |
| 7 | Notebook §6 → **Airflow → MinIO** | el gráfico del brillo con el tráfico recién enviado: normal sin drift, de noche con drift. Después disparar `drift_monitoring` (la tarea `alert` en rojo) y abrir la alerta en `monitoring/alerts/` | "Sin etiquetas se vigilan las condiciones de captura; el DAG lo hace cada hora y alerta a una persona." |
| 8 | Notebook §7 → **Airflow → MLflow** | en Airflow, `register_model` → Trigger con los valores por defecto (`model-v1`); el notebook espera y muestra la tabla: v2 `champion` (mAP 0.647) y v1 `previous` (0.545), con el motivo. El gateway pasa a servir v2 solo y la misma foto se ve con v1 y con v2. En MLflow, las dos versiones con sus alias | "Llega un modelo entrenado con todo el dataset: el gate lo mide contra el champion sobre el mismo set, lo promueve, y el servicio lo carga sin reiniciar. El anterior queda como `previous` para el rollback." |
| 9 | Notebook §8 | se altera un archivo de los pesos del champion en MinIO y se reinicia inference: el log dice `IntegrityError` y `/health` da 503; se restaura y vuelve a servir | "El hash registrado al promover manda: un peso alterado no se sirve." |

Duración: ~9 min (las esperas de los DAGs, de los recorridos y del reinicio se cortan al editar). El §4–§6 del notebook tienen que
correr seguidos: el drift usa el tráfico de esos dos recorridos.

## Si algo sale mal

- Sin ventanas en el §4: `docker logs billboard-detector-mlops-stream-consumer-1` (el consumidor tiene que estar corriendo).
- La demo escribe en `predictions/`: el `drift_monitoring` de la hora siguiente ve ese tráfico (el de noche da drift). Para que el DAG muestre drift, disparalo después del §5 del notebook.
