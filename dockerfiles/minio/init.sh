#!/bin/sh
# Buckets y un usuario de MinIO por servicio con mínimo privilegio (docs/seguridad.md).
# Airflow y MLflow usan la cuenta root (administran el lake y los artefactos).
set -e
mc alias set s3 http://s3:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "s3/$MLFLOW_BUCKET"
mc mb --ignore-existing "s3/$BUCKET"

resources() {  # prefijos → "arn:...prefijo*","arn:...prefijo2*"
  out=""; for p in $1; do out="$out\"arn:aws:s3:::$BUCKET/$p*\","; done; echo "${out%,}"
}

policy() {  # policy <nombre> <prefijos de lectura> <prefijos de escritura>
  stmts="{\"Effect\": \"Allow\", \"Action\": [\"s3:ListBucket\"], \"Resource\": [\"arn:aws:s3:::$BUCKET\"]}"
  [ -n "$2" ] && stmts="$stmts, {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\"], \"Resource\": [$(resources "$2")]}"
  [ -n "$3" ] && stmts="$stmts, {\"Effect\": \"Allow\", \"Action\": [\"s3:PutObject\"], \"Resource\": [$(resources "$3")]}"
  echo "{\"Version\": \"2012-10-17\", \"Statement\": [$stmts]}" > "/tmp/$1.json"
  mc admin policy create s3 "$1" "/tmp/$1.json"
}

user() {  # user <nombre> <secreto> <política>
  mc admin user add s3 "$1" "$2"
  mc admin policy attach s3 "$3" --user "$1" 2>/dev/null || true
}

# inference: lee los pesos; escribe predicciones
policy inference-rw "models/" "predictions/"
# consumidor de streaming: escribe métricas por ventana y alertas
policy stream-w "" "curated/stream_windows/ monitoring/alerts/"

user inference "$INFERENCE_S3_SECRET" inference-rw
user stream "$STREAM_S3_SECRET" stream-w
echo "buckets y usuarios listos"
