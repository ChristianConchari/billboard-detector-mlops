#!/usr/bin/env bash
# Demo guiada de VallaVision para grabar la presentación.
#
#   scripts/demo.sh            # todos los segmentos, con pausa (Enter) antes de cada comando
#   scripts/demo.sh 3 6        # solo esos segmentos (para grabar por partes)
#   scripts/demo.sh --lista    # lista de segmentos
#   DEMO_SIN_PAUSAS=1 scripts/demo.sh 4   # sin pausas (ensayo)
#   DEMO_CLI=1 scripts/demo.sh 2          # dispara los DAGs por comando en vez de esperar la interfaz
#
# Lo que se ve mejor en una interfaz va en el navegador: los DAGs se disparan desde Airflow
# (segmentos 2 y 6; el script espera la corrida y resume en la terminal) y la inferencia se prueba
# en /docs del gateway (segmento 3). El resto queda en la terminal.
#
# Requisitos: stack arriba y con champion (README, Instalación). Los secretos se leen de .env y
# nunca se imprimen: en pantalla aparecen como $GATEWAY_API_KEYS, $INFERENCE_TOKEN, etc.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

DEMO_IMG=${DEMO_IMG:-data/billboards-2025-09-14-sur-lp/images/042e2f0b-229c-46b6-8236-1e3391cb3682_1647.jpg}
VENV=$ROOT/.venv-demo
PY=$VENV/bin/python
INFERENCE=billboard-detector-mlops-inference-1
CONSUMER=billboard-detector-mlops-stream-consumer-1
WORKER=billboard-detector-mlops-airflow-worker-1

SEGMENTOS=(
  "1|Stack: un docker compose levanta todo"
  "2|Registro y promoción: gate, linaje y alias en MLflow"
  "3|Inferencia REST con las cajas dibujadas"
  "4|Seguridad: API key, formato real y token gRPC"
  "5|Streaming con Redpanda: métricas por ventana de event time"
  "6|Monitoreo de drift: reporte y alerta"
  "7|Integridad: pesos alterados no se sirven"
)

# --- presentación ---------------------------------------------------------------------
B=$'\e[1m'; C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; D=$'\e[2m'; R=$'\e[0m'

titulo() { clear; printf "\n${B}${C}━━ %s ━━${R}\n\n" "$1"; }
nota() { printf "${D}# %s${R}\n" "$*"; }
pausa() { [ -n "${DEMO_SIN_PAUSAS:-}" ] && return; read -rsp "" _; }
# corre: muestra el comando tal cual (con las variables sin expandir) y lo ejecuta
corre() { printf "${G}\$${R} ${B}%s${R}" "$1"; pausa; echo; eval "$1"; echo; }
ok() { printf "${G}✔ %s${R}\n" "$*"; }

preparar() {
  [ -f .env ] || { echo "falta .env (scripts/generar_env.sh)"; exit 1; }
  set -a; . ./.env; set +a
  if [ ! -x "$PY" ]; then
    echo "preparando $VENV (una sola vez)…"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" -q install pillow==12.3.0 grpcio==1.84.0 protobuf==6.33.6 grpcio-tools==1.81.1 kafka-python==3.0.11
  fi
  [ -f vallavision/grpc_gen/detector_pb2_grpc.py ] || PYTHON=$PY sh scripts/gen_proto.sh
  # Un DAG pausado acepta el disparo desde la interfaz pero no corre: se despausan acá
  for dag in registrar_modelo monitoreo_drift rollback_modelo; do
    docker exec airflow-apiserver airflow dags unpause "$dag" >/dev/null 2>&1
  done
  if ! curl -sf localhost:${FASTAPI_PORT:-8800}/health >/dev/null; then
    echo "el gateway no tiene champion cargado: correr registrar_modelo antes de grabar"; exit 1
  fi
}

esperar_dag() {  # esperar_dag <dag> <run_id>: muestra las tareas a medida que terminan
  local dag=$1 run=$2 vistos="" estado terminado=""
  while true; do
    local hechas
    hechas=$(docker exec airflow-apiserver airflow tasks states-for-dag-run "$dag" "$run" -o json 2>/dev/null |
      python3 -c "import json,sys;[print(t['task_id'],t['state']) for t in json.load(sys.stdin) if t['state'] in ('success','failed','upstream_failed')]" 2>/dev/null)
    while read -r tarea st; do
      [ -z "$tarea" ] && continue
      case " $vistos " in *" $tarea "*) ;; *) vistos="$vistos $tarea"
        [ "$st" = success ] && printf "  ${G}✔${R} %s\n" "$tarea" || printf "  ${Y}✘${R} %s (%s)\n" "$tarea" "$st";; esac
    done <<< "$hechas"
    case "${terminado:-}" in 1) printf "\nrun: ${B}%s${R}\n" "$estado"; return;; esac
    estado=$(docker exec airflow-apiserver airflow dags state "$dag" "$run" 2>/dev/null | tail -1)
    # una vuelta más después del estado final, para mostrar las últimas tareas
    case "$estado" in *success*|*failed*) terminado=1; continue;; esac
    sleep 5
  done
}

disparar() {  # disparar <dag> → run_id
  docker exec airflow-apiserver airflow dags trigger "$1" -o json 2>/dev/null |
    python3 -c "import json,sys;print(json.load(sys.stdin)[0]['dag_run_id'])"
}

ultimo_run_manual() {  # último run disparado a mano (los horarios de monitoreo_drift no cuentan)
  docker exec airflow-apiserver airflow dags list-runs "$1" -o json 2>/dev/null | python3 -c "
import json, sys
runs = [r for r in json.load(sys.stdin) if r['run_id'].startswith('manual__')]
runs.sort(key=lambda r: r.get('run_after') or r.get('start_date') or '')
print(runs[-1]['run_id'] if runs else '')"
}

lanzar() {  # lanzar <dag> → run_id: espera el disparo desde la interfaz de Airflow (o lo dispara con DEMO_CLI=1)
  local dag=$1
  if [ -n "${DEMO_CLI:-}" ]; then
    printf "${G}\$${R} ${B}docker exec airflow-apiserver airflow dags trigger %s${R}\n" "$dag" >&2
    disparar "$dag"; return
  fi
  local antes nuevo; antes=$(ultimo_run_manual "$dag")
  printf "${Y}▶ En Airflow (http://localhost:%s): Dags → %s → Trigger → Trigger, y abrir la vista Graph${R}\n" \
    "${AIRFLOW_PORT:-8080}" "$dag" >&2
  printf "${D}  esperando la corrida…${R}\n" >&2
  while true; do
    nuevo=$(ultimo_run_manual "$dag")
    [ -n "$nuevo" ] && [ "$nuevo" != "$antes" ] && { echo "$nuevo"; return; }
    sleep 3
  done
}

esperar_inference() {  # esperar_inference <healthy|unhealthy>
  for _ in $(seq 1 60); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' $INFERENCE 2>/dev/null)" = "$1" ] && return 0
    sleep 3
  done
  return 1
}

# --- segmentos -------------------------------------------------------------------------

seg1() {
  titulo "1 · Stack: Airflow, MLflow, MinIO, Postgres, Redpanda, gateway e inferencia"
  corre 'docker compose --profile all ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"'
  nota "solo el gateway escucha fuera de la máquina; lo demás está en 127.0.0.1"
  nota "Airflow http://localhost:${AIRFLOW_PORT:-8080} · MLflow http://localhost:${MLFLOW_PORT:-5001} · MinIO http://localhost:${MINIO_PORT_UI:-9001}"
  pausa
}

seg2() {
  titulo "2 · Registro y promoción del modelo (DAG registrar_modelo)"
  nota "baja model-v1 del release (SHA-256), mide el mAP con IC sobre el test revisado,"
  nota "re-evalúa al champion sobre el mismo set y decide el alias"
  local run; run=$(lanzar registrar_modelo); echo "run: $run"; echo
  esperar_dag registrar_modelo "$run"
  echo
  nota "lo que quedó en MLflow: alias, métricas del gate con IC y linaje"
  printf "${G}\$${R} ${B}resumen del registro en MLflow${R}"; pausa; echo
  docker exec -i airflow-apiserver python - <<'PY'
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://mlflow:5000")
c = MlflowClient()
for found in sorted(c.search_model_versions("name='billboard-detector'"), key=lambda v: int(v.version)):
    mv = c.get_model_version("billboard-detector", found.version)  # la búsqueda no trae los alias
    t = mv.tags
    m = c.get_run(t["eval_run_id"]).data.metrics
    print(f"versión {mv.version}  alias {mv.aliases}")
    print(f"  gate     {t['eval_set']}: mAP {m['gate_map']:.3f}  IC 95 % [{m['gate_map_ci_low']:.3f}, {m['gate_map_ci_high']:.3f}]  AP50 {m['gate_map_50']:.3f}")
    print(f"  motivo   {t.get('promotion_reason', '-')}")
    print(f"  pesos    {t['weights_uri']}  sha256 {t['weights_dir_sha256'][:16]}…")
    print(f"  código   commit {t['git_commit'][:12]}  code_sha256 {t['code_sha256'][:16]}…")
PY
  echo
  nota "en MLflow (http://localhost:${MLFLOW_PORT:-5001}): experimento evaluaciones → último run →"
  nota "métricas gate_map e IC, tag git_commit y el artefacto eval_set/test-….json; Models → billboard-detector (alias champion)"
  pausa
}

seg3() {
  titulo "3 · Inferencia REST: POST /v1/detect"
  local url="http://localhost:${FASTAPI_PORT:-8800}/docs"
  # La clave va al portapapeles, nunca a la pantalla
  if command -v xclip >/dev/null; then
    printf %s "${GATEWAY_API_KEYS%%,*}" | xclip -selection clipboard && ok "API key copiada al portapapeles"
  elif command -v wl-copy >/dev/null; then
    printf %s "${GATEWAY_API_KEYS%%,*}" | wl-copy && ok "API key copiada al portapapeles"
  fi
  printf "${Y}▶ En %s: Authorize → pegar la clave → POST /v1/detect → Try it out → elegir una imagen → Execute${R}\n" "$url"
  [ -z "${DEMO_NO_ABRIR:-}" ] && (xdg-open "$url" >/dev/null 2>&1 &)
  nota "imagen sugerida: $DEMO_IMG"
  pausa
  nota "la misma llamada, con las cajas dibujadas sobre la foto"
  corre "$PY scripts/ver_detecciones.py $DEMO_IMG${DEMO_NO_ABRIR:+ --no-abrir}"
  nota "la predicción quedó en s3://data/predictions/ (sin la imagen)"
  pausa
}

seg4() {
  titulo "4 · Seguridad"
  local url="localhost:${FASTAPI_PORT:-8800}"
  nota "sin API key / con una clave inválida"
  corre "curl -s -w '  → %{http_code}\n' $url/v1/model"
  corre "curl -s -w '  → %{http_code}\n' -H 'X-API-Key: no-es-mia' $url/v1/model"
  nota "un GIF renombrado .jpg y declarado image/jpeg: se mira el formato real"
  $PY -c "from PIL import Image; Image.new('RGB',(64,64),'red').save('/tmp/vallavision-falsa.jpg','GIF')"
  corre "curl -s -w '  → %{http_code}\n' -H \"X-API-Key: \$GATEWAY_API_KEYS\" -F 'image=@/tmp/vallavision-falsa.jpg;type=image/jpeg' $url/v1/detect"
  nota "recargar el modelo exige clave admin"
  corre "curl -s -w '  → %{http_code}\n' -X POST -H \"X-API-Key: \$GATEWAY_API_KEYS\" $url/v1/reload"
  nota "llamar al modelo por gRPC saltando el gateway, sin token"
  corre "$PY -c \"import grpc,sys; sys.path.insert(0,'.'); from vallavision.grpc_gen import detector_pb2 as pb, detector_pb2_grpc as g
try: g.DetectorStub(grpc.insecure_channel('localhost:50051')).GetModel(pb.GetModelRequest())
except grpc.RpcError as e: print('gRPC sin token →', e.code().name)\""
  pausa
}

ventanas() {  # muestra como tabla las ventanas que cierra el consumidor; corta al matar `docker logs`
  docker logs -f --since 1s $CONSUMER 2>&1 & echo $! > /tmp/vallavision-demo-logs.pid
}
formatear() {
  python3 -u -c '
import json, sys, time
for line in sys.stdin:
    if line.startswith("ALERTA"):
        print("\033[33m⚠ " + line.strip() + "\033[0m", flush=True); continue
    if not line.startswith("{"): continue
    r = json.loads(line)
    s, e = (time.strftime("%H:%M:%S", time.localtime(t / 1000)) for t in r["window"])
    print("cámara %s  %s–%s  %2d frames  %.1f fps  p95 %4.0f ms  %.1f vallas/frame  %d tardíos" % (
        r["camera"], s, e, r["frames"], r["throughput_fps"], r["latency_ms_p95"] or 0,
        r["detections_per_frame"], r["late_frames"]), flush=True)
'
}
stream() {  # stream <título> <args del productor>
  nota "$1"
  printf "${G}\$${R} ${B}python scripts/camara_productor.py %s${R}" "$2"; pausa; echo
  { ventanas; wait; } | formatear & local f=$!
  # shellcheck disable=SC2086
  $PY scripts/camara_productor.py $2 2>&1 | sed "s/^/${D}productor: /;s/$/${R}/"
  sleep 15  # el consumidor cierra las ventanas de una cámara tras 10 s sin frames
  kill "$(cat /tmp/vallavision-demo-logs.pid)" 2>/dev/null; wait $f 2>/dev/null
  echo
}

seg5() {
  titulo "5 · Streaming: 3 cámaras → Redpanda → consumidor → ventanas de 15 s por event time"
  stream "cámaras del test revisado (videos que no están en la referencia de drift)" "--rate 2 --loops 1"
  stream "las mismas cámaras de noche (frames oscurecidos): lo va a ver monitoreo_drift" "--rate 2 --loops 1 --degrade oscurecer"
  nota "el topic: 3 particiones; la cámara es la clave, así que sus frames caen siempre en la misma (por hash)"
  corre "docker exec redpanda rpk topic describe frames -p"
  nota "el grupo de consumidores: qué offset confirmó y cuánto le falta (lag)"
  corre "docker exec redpanda rpk group describe detector"
  pausa
}

seg6() {
  titulo "6 · Monitoreo de drift (DAG monitoreo_drift, horario)"
  nota "compacta la última hora de predicciones, REST y streaming (solo umbral calibrado): chi² + KS con Holm"
  local run; run=$(lanzar monitoreo_drift); echo "run: $run"; echo
  esperar_dag monitoreo_drift "$run"
  nota "en Airflow: la tarea alertar queda en rojo a propósito (DriftDetected); su log lo explica"
  echo; printf "${G}\$${R} ${B}lo que dejó en el Data Lake${R}"; pausa; echo
  docker exec -i $WORKER python - <<'PY'
import boto3, json
s3 = boto3.client("s3", endpoint_url="http://s3:9000")
def last(prefix):
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket="data", Prefix=prefix).get("Contents", []))
    return (keys[-1], json.loads(s3.get_object(Bucket="data", Key=keys[-1])["Body"].read())) if keys else (None, None)
k, r = last("monitoring/drift/")
print(f"reporte  s3://data/{k}: {r['n_frames']} frames, drift={r['drift']}")
for f, m in r["metrics"].items():
    if "p_value" in m:
        print(f"   {f:13} {m['test']:4} ref {m['ref_mean']:.3f} → {m['cur_mean']:.3f}   p={m['p_value']:.2g}   {'DRIFT' if m['drift'] else 'ok'}")
k, a = last("monitoring/alerts/")
print(f"alerta   s3://data/{k}: [{a['kind']}] {a['message']}")
PY
  echo
  nota "en MinIO (http://localhost:${MINIO_PORT_UI:-9001}): bucket data → monitoring/alerts/ y monitoring/drift/"
  pausa
}

seg7() {
  titulo "7 · Integridad: alguien modifica los pesos en el Data Lake"
  local key=models/billboard-detector/v1/calibration.json
  restaurar() {
    docker exec $WORKER python -c "
import boto3; s=boto3.client('s3',endpoint_url='http://s3:9000')
s.put_object(Bucket='data',Key='$key',Body=open('/tmp/calib.demo','rb').read())" && docker restart $INFERENCE >/dev/null
  }
  docker exec $WORKER python -c "
import boto3; s=boto3.client('s3',endpoint_url='http://s3:9000')
open('/tmp/calib.demo','wb').write(s.get_object(Bucket='data',Key='$key')['Body'].read())"
  trap 'restaurar; trap - INT' INT
  printf "${G}\$${R} ${B}umbral 0.99 escrito sobre s3://data/$key${R}"; pausa; echo
  docker exec $WORKER python -c "
import boto3; s=boto3.client('s3',endpoint_url='http://s3:9000')
s.put_object(Bucket='data',Key='$key',Body=b'{\"score_threshold\": 0.99}')"
  corre "docker restart $INFERENCE"
  esperar_inference unhealthy || sleep 30
  corre "docker logs --tail 2 $INFERENCE"
  corre "curl -s -w '  → %{http_code}\n' localhost:${FASTAPI_PORT:-8800}/health"
  nota "restaurando los pesos originales…"
  restaurar; trap - INT
  esperar_inference healthy && ok "pesos verificados de nuevo: inference sirve el champion"
  corre "curl -s -w '  → %{http_code}\n' localhost:${FASTAPI_PORT:-8800}/health"
  pausa
}

# --- main ------------------------------------------------------------------------------
if [ "${1:-}" = "--lista" ]; then
  for s in "${SEGMENTOS[@]}"; do printf "%s  %s\n" "${s%%|*}" "${s#*|}"; done; exit 0
fi
preparar
for n in ${@:-1 2 3 4 5 6 7}; do
  declare -F "seg$n" >/dev/null || { echo "segmento desconocido: $n (ver --lista)"; exit 1; }
  "seg$n"
done
titulo "Fin"
