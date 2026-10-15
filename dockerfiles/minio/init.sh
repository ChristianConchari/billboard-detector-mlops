#!/bin/sh
# Buckets y un usuario de MinIO por servicio con mínimo privilegio (docs/seguridad.md).
# Airflow y MLflow usan la cuenta root (administran el lake y los artefactos).
set -e
mc alias set s3 http://s3:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "s3/$MLFLOW_BUCKET"
mc mb --ignore-existing "s3/$BUCKET"

policy() {  # policy <nombre> <prefijos de lectura> <prefijos de escritura>
  read_res=""; for p in $2; do read_res="$read_res\"arn:aws:s3:::$BUCKET/$p*\","; done
  write_res=""; for p in $3; do write_res="$write_res\"arn:aws:s3:::$BUCKET/$p*\","; done
  cat > "/tmp/$1.json" <<JSON
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": ["arn:aws:s3:::$BUCKET"]},
  {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [${read_res%,}]}
  $( [ -n "$3" ] && echo ", {\"Effect\": \"Allow\", \"Action\": [\"s3:PutObject\"], \"Resource\": [${write_res%,}]}" )
]}
JSON
  mc admin policy create s3 "$1" "/tmp/$1.json"
}

user() {  # user <nombre> <secreto> <política>
  mc admin user add s3 "$1" "$2"
  mc admin policy attach s3 "$3" --user "$1" 2>/dev/null || true
}

# inference: lee pesos y referencias; escribe predicciones y la muestra de imágenes
policy inference-rw "models/ curated/reference/" "predictions/ raw/stream/"
# gateway (GraphQL): solo lee predicciones
policy gateway-ro "predictions/" ""
# consumidor de streaming: lee referencias; escribe ventanas y alertas
policy stream-rw "curated/reference/" "curated/stream_windows/ monitoring/alerts/"

user inference "$INFERENCE_S3_SECRET" inference-rw
user gateway "$GATEWAY_S3_SECRET" gateway-ro
user stream "$STREAM_S3_SECRET" stream-rw
echo "buckets y usuarios listos"
