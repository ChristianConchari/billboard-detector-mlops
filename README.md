# billboard-detector-mlops

**VallaVision**: detector de vallas publicitarias en imágenes de calle, puesto en producción.
TP integrador de *Operaciones de Aprendizaje Automático II* (CEIA · FIUBA). Autor: Christian Conchari.

El foco es la **puesta en producción**, no el modelado: el detector entra como artefacto ya entrenado,
un RT-DETR (ResNet-50) de clase única publicado en el release
[`model-v1`](https://github.com/ChristianConchari/vit-billboard-detector/releases/tag/model-v1)
(mAP 0.649 · AP50 0.930 en test, umbral calibrado 0.15).

- [docs/alcance.md](docs/alcance.md): alcance, arquitectura, decisiones, riesgos e hitos
- [docs/contrato.md](docs/contrato.md): contrato de entrada/salida, layout del Data Lake y DAGs

## Servicios

Base: [facundolucianna/amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) (remoto `upstream`).

| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 (airflow / airflow) |
| MLflow | http://localhost:5001 |
| MinIO | http://localhost:9001 (minio / minio123) |
| API REST de detección | http://localhost:8800/docs |
| API gRPC de detección | `localhost:50051` (`proto/detector.proto`) |
| Label Studio (`--profile labeling`) | http://localhost:8090 (admin@vallavision.local / labelstudio) |

Buckets: `data` (Data Lake por zonas, ver contrato §4) y `mlflow` (artefactos de MLflow).

## Dataset y revisión de etiquetas

Dataset `billboards-2025-09-14-sur-lp` (export COCO de origen, clase única `billboard`):
632 imágenes de 26 videos, split por video (train 406 / val 169 / test 57). Las imágenes
no se versionan en git: descomprimir el zip en `data/`, que queda como
`data/billboards-2025-09-14-sur-lp/{annotations,images,manifest.json}`.

El export incluye cajas sin revisar (`verdict: pending`). Para revisarlas con Label Studio:

```bash
docker compose --profile labeling up -d
python scripts/revision_etiquetas.py importar   # proyecto + imágenes con cajas pendientes
# en http://localhost:8090: dejar la caja = confirmar, moverla = ajustar, borrarla = rechazar; Submit
python scripts/revision_etiquetas.py exportar   # -> data/<dataset>/annotations_revisadas/
```

Las cajas rechazadas quedan con `iscrowd: 1` (pycocotools las ignora al evaluar). La
revisión se guarda en el volumen `label_studio_data`, así que se puede exportar varias veces.

## Uso rápido

```bash
docker compose --profile all up -d --build
# Airflow → DAG registrar_modelo → Trigger: baja model-v1 del release (verifica SHA-256),
# lo sube a MinIO, evalúa mAP sobre val, lo registra en MLflow y lo promueve a `champion`
curl -F image=@data/billboards-2025-09-14-sur-lp/images/<imagen>.jpg http://localhost:8800/v1/detect
```

Sin `champion` la API responde `503` y lo busca cada 30 s; `POST /v1/reload` fuerza la recarga.
Cada predicción se guarda en `s3://data/predictions/date=…/<request_id>.json`.

gRPC (`Detect` y `DetectStream`) y comparación contra REST:

```bash
scripts/gen_proto.sh   # genera vallavision/grpc_gen/ (requiere grpcio-tools)
python scripts/bench_rest_vs_grpc.py <imagen.jpg> -n 50
```

Streaming simulado y drift: `scripts/stream_client.py` reproduce una carpeta de imágenes como cámara
(un `Detect` por frame) y reporta por ventana throughput, latencia p50/p95 y detecciones por frame; con
`--reference` también compara contra la referencia (KS) y `--degrade oscurecer` provoca drift para la demo.
El DAG `monitoreo_drift` (horario) hace la misma comparación sobre `predictions/` y falla si hay drift;
la referencia la genera `registrar_modelo` sobre `val` en `curated/reference/`.

```bash
uv run --with grpcio --with pillow --with numpy --with scipy --with boto3 scripts/stream_client.py \
  data/billboards-2025-09-14-sur-lp/images/ --fps 5 --degrade oscurecer
```

Tests de la API y de drift: `pytest tests` (usan un detector falso, no hace falta el stack).

## Estructura

| Ruta | Qué hay |
|---|---|
| `vallavision/` | Paquete compartido: `detector.py` (RT-DETR + pyfunc de MLflow), `service.py` (carga del champion, validación, persistencia), `grpc_server.py` |
| `vallavision/drift.py` | Features por frame, referencia y test KS |
| `airflow/dags/registrar_modelo.py` | DAG de registro y promoción del modelo (genera también la referencia de drift) |
| `airflow/dags/monitoreo_drift.py` | DAG horario: `predictions/` vs referencia → reporte en `monitoring/drift/` |
| `scripts/stream_client.py` | Cámara simulada + métricas por ventana por gRPC |
| `dockerfiles/fastapi/` | Imagen de REST y gRPC (contexto de build: raíz del repo) |
| `proto/detector.proto` | Contrato gRPC |

## Estado

| Paso | Estado |
|---|---|
| Stack base (Airflow, MLflow, MinIO, Postgres, FastAPI) levanta healthy | ✅ |
| Contrato de entrada/salida y layout del Data Lake | ✅ borrador |
| DAG `registrar_modelo` (RT-DETR → MinIO → MLflow → referencia de drift) | ✅ corre de punta a punta; ⚠️ la promoción a `champion` está bloqueada por el mAP (ver abajo) |
| REST `/v1/detect`, `/v1/model`, `/health` | ✅ probado con el RT-DETR real |
| gRPC `Detect` + benchmark vs REST | ✅ probado (`DetectStream` sin probar) |
| Streaming + drift (`stream_client.py`), DAG `monitoreo_drift` | ✅ sin drift con tráfico normal; con `--degrade oscurecer` marca brillo y contraste |

### ⚠️ mAP de `registrar_modelo` no comparable con el del release

Sobre `val` de este dataset el DAG mide **mAP 0.066 / AP50 0.24**, contra 0.817 (val) y 0.649 (test) del
release. La inferencia es la misma que la del repo del modelo y las detecciones se ven bien, así que no es un
fallo de carga. La diferencia está en la evaluación: (1) el split por video de este dataset no coincide con el
del entrenamiento (2 de sus 4 videos de `val` eran de train del modelo) y (2) 102 de las 169 cajas de `val` son
pre-etiquetas `pending`, mientras que el modelo se evaluó con cajas revisadas a mano. Mientras no se evalúe
con anotaciones revisadas, `min_map` (0.5) deja al modelo como `challenger`; para probar el serving se fijó
`champion` a mano con `set_registered_model_alias`.

## Instalación

1. Para poder levantar todos los servicios, primero instala [Docker](https://docs.docker.com/engine/install/) en tu computadora (o en el servidor que desees usar).
2. Clona este repositorio.
3. Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no podrás subir DAGs (en `airflow/dags`) o plugins, etc.
4. En la carpeta raíz de este repositorio, ejecuta:

```bash
docker compose --profile all up
```

5. Una vez que todos los servicios estén funcionando (verifica con el comando `docker ps -a` que todos los servicios estén healthy o revisa en Docker Desktop), podrás acceder a los diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

Si estás usando un servidor externo a tu computadora de trabajo, reemplaza `localhost` por su IP (puede ser una privada si tu servidor está en tu LAN o una IP pública si no; revisa firewalls u otras reglas que eviten las conexiones).

Todos los puertos u otras configuraciones se pueden modificar en el archivo `.env`. Se invita a jugar y romper para aprender; siempre puedes volver a clonar este repositorio.

## Apagar los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no se están utilizando, se recomienda detenerlos. Para hacerlo, ejecuta el siguiente comando:

```bash
docker compose --profile all down
```

Si deseas no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), utiliza el siguiente comando:

```bash
docker compose down --rmi all --volumes
```

Nota: Si haces esto, perderás todo en los buckets y bases de datos.

## Aspectos específicos de Airflow

### Variables de entorno
Airflow ofrece una amplia gama de opciones de configuración. En el archivo `docker-compose.yaml`, dentro de `x-airflow-common`, se encuentran variables de entorno que pueden modificarse para ajustar la configuración de Airflow. Pueden añadirse [otras variables](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html).

### Uso de ejecutores externos
Actualmente, para este caso, Airflow utiliza un ejecutor [celery](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html), lo que significa que las tareas se ejecutan en otro contenedor. 

### Uso de la CLI de Airflow

Si necesitan depurar Apache Airflow, pueden utilizar la CLI de Apache Airflow de la siguiente manera:

```bash
docker compose --profile all --profile debug up
```

Una vez que el contenedor esté en funcionamiento, pueden utilizar la CLI de Airflow de la siguiente manera, 
por ejemplo, para ver la configuración:

```bash
docker-compose run airflow-cli config list      
```

Para obtener más información sobre el comando, pueden consultar [aqui](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html).

### Variables y Conexiones

Si desean agregar variables para accederlas en los DAGs, pueden hacerlo en `secrets/variables.yaml`. Para obtener más [información](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html), 
consulten la documentación.

Si desean agregar conexiones en Airflow, pueden hacerlo en `secrets/connections.yaml`. También es posible agregarlas mediante la interfaz de usuario (UI), pero estas no persistirán si se borra todo. Por otro lado, cualquier conexión guardada en `secrets/connections.yaml` no aparecerá en la UI, aunque eso no significa que no exista. Consulten la documentación para obtener más 
[información](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html).

## Conexión con los buckets

Dado que no estamos utilizando Amazon S3, sino una implementación local de los mismos mediante MinIO, es necesario modificar las variables de entorno para conectar con el servicio de MinIO. Las variables de entorno son las siguientes:

```bash
AWS_ACCESS_KEY_ID=minio   
AWS_SECRET_ACCESS_KEY=minio123 
AWS_ENDPOINT_URL_S3=http://localhost:90000
```

MLflow también tiene una variable de entorno que afecta su conexión a los buckets:

```bash
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```
Asegúrate de establecer estas variables de entorno antes de ejecutar tu notebook o scripts en tu máquina o en cualquier otro lugar. Si estás utilizando un servidor externo a tu computadora de trabajo, reemplaza localhost por su dirección IP.

Al hacer esto, podrás utilizar `boto3`, `awswrangler`, etc., en Python con estos buckets, o `awscli` en la consola.

Si tienes acceso a AWS S3, ten mucho cuidado de no reemplazar tus credenciales de AWS. Si usas las variables de entorno, no tendrás problemas.

## Valkey

La base de datos Valkey es usada por Apache Airflow para su funcionamiento. Tal como está configurado ahora no esta expuesto el puerto para poder ser usado externamente. Se puede modificar el archivo `docker-compose.yaml` para habilitaro.

## Pull Request

Este repositorio está abierto para que realicen sus propios Pull Requests y así contribuir a mejorarlo. Si desean realizar alguna modificación, **¡son bienvenidos!** También se pueden crear nuevos entornos productivos para aumentar la variedad de implementaciones, idealmente en diferentes `branches`. Algunas ideas que se me ocurren que podrían implementar son:

- Reemplazar Airflow y MLflow con [Metaflow](https://metaflow.org/) o [Kubeflow](https://www.kubeflow.org).
- Reemplazar MLflow con [Seldon-Core](https://github.com/SeldonIO/seldon-core).
- Agregar un servicio de tableros como, por ejemplo, [Grafana](https://grafana.com).
