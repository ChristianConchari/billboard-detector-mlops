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

## Antes de grabar

- [ ] Stack arriba y con champion (`curl localhost:8800/health` → `model_loaded: true`). No construir en cámara.
- [ ] `.env` propio para la grabación (`mv .env .env.real && scripts/generate_env.sh` y `down -v` + `up` + `register_model`), o al menos no abrirlo nunca en pantalla.
- [ ] Sin benchmarks en la hora previa: contaminan el drift que ve `drift_monitoring`.
- [ ] Cerrar navegador pesado y otros proyectos: el grabador y el modelo compiten por la CPU y la latencia sube.
- [ ] JupyterLab con zoom ≥ 125 % y el notebook sin salidas (Kernel → Restart Kernel and Clear Outputs).
- [ ] Pestañas abiertas y con la sesión iniciada: Airflow, MLflow, MinIO.
- [ ] Ensayo: correr el notebook entero una vez (≈ 4 min); deja el sistema como estaba.

## Orden de la grabación

| # | Dónde | Qué se ve | Qué decir |
|---|---|---|---|
| 1 | Notebook, intro + celda 1 | el caso de uso (tabla de piezas y su papel en la plataforma) y el modelo servido: versión, mAP del gate con IC | "El sistema es el servicio de detección de una plataforma de etiquetado geolocalizado. Esto es lo que está sirviendo." |
| 2 | **Airflow → MLflow** | disparar `register_model` (Trigger, vista Graph; cortar la espera) → log de `promote` → en MLflow el run de `evaluations` (métricas, `git_commit`, artefacto `eval_set`) y el modelo con alias `champion` | "Ningún modelo nuevo cambia las etiquetas de la plataforma sin pasar un gate sobre datos que no vio, con linaje completo." |
| 3 | Notebook §1 | la foto con las 5 vallas y su tabla; el mosaico de 4 recorridos con la posición GPS sobreimpresa | "La plataforma manda un frame; el servicio devuelve las cajas y la versión del modelo que las produjo." |
| 4 | Notebook §2 | REST vs gRPC: bytes y latencia; base64 +33 % | "Adentro, gRPC: bytes tal cual, contrato tipado. La plataforma solo ve el gateway." |
| 5 | Notebook §3 | la tabla de ataques, todo ✔ | "Los controles de seguridad, contra el sistema real." |
| 6 | Notebook §4–5 | los vehículos publican en Redpanda; ventanas por recorrido; particiones del topic; después de noche | "El broker absorbe la diferencia entre lo que producen los vehículos y lo que infiere el modelo; las ventanas son por instante de captura." |
| 7 | Notebook §6 → **Airflow → MinIO** | el gráfico del brillo con el tráfico recién enviado: normal sin drift, de noche con drift. Después disparar `drift_monitoring` (la tarea `alert` en rojo) y abrir la alerta en `monitoring/alerts/` | "Sin etiquetas se vigilan las condiciones de captura; el DAG lo hace cada hora y alerta a una persona." |
| 8 | Notebook §7 → **Airflow → MLflow** | el notebook prepara un candidato (los mismos pesos con otro umbral) en `staging/`; en Airflow, `register_model` con ese `weights_source`; el notebook espera y muestra la tabla: v1 `champion`, v2 `challenger` con el motivo "Δ +0.0000 < 0.01". En MLflow, las dos versiones | "Un candidato que no mejora al champion sobre el mismo set no llega a producción, y el motivo queda escrito." |
| 9 | Notebook §8 | se altera un archivo de los pesos del champion en MinIO y se reinicia inference: el log dice `IntegrityError` y `/health` da 503; se restaura y vuelve a servir | "El hash registrado al promover manda: un peso alterado no se sirve." |

Duración: ~9 min (las esperas de los DAGs, de los recorridos y del reinicio se cortan al editar). El §4–§6 del notebook tienen que
correr seguidos: el drift usa el tráfico de esos dos recorridos.

## Si algo sale mal

- Sin ventanas en el §4: `docker logs billboard-detector-mlops-stream-consumer-1` (el consumidor tiene que estar corriendo).
- La demo escribe en `predictions/`: el `drift_monitoring` de la hora siguiente ve ese tráfico (el de noche da drift). Para que el DAG muestre drift, disparalo después del §5 del notebook.
