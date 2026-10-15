#!/bin/sh
# Buckets plus one MinIO user per service with least privilege (docs/seguridad.md).
# Airflow and MLflow use the root account (they manage the data lake and the artifacts).
set -e
mc alias set s3 http://s3:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "s3/$MLFLOW_BUCKET"
mc mb --ignore-existing "s3/$BUCKET"

resources() {  # prefixes → "arn:...prefix*","arn:...prefix2*"
  out=""; for p in $1; do out="$out\"arn:aws:s3:::$BUCKET/$p*\","; done; echo "${out%,}"
}

policy() {  # policy <name> <read prefixes> <write prefixes>
  stmts="{\"Effect\": \"Allow\", \"Action\": [\"s3:ListBucket\"], \"Resource\": [\"arn:aws:s3:::$BUCKET\"]}"
  [ -n "$2" ] && stmts="$stmts, {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\"], \"Resource\": [$(resources "$2")]}"
  [ -n "$3" ] && stmts="$stmts, {\"Effect\": \"Allow\", \"Action\": [\"s3:PutObject\"], \"Resource\": [$(resources "$3")]}"
  echo "{\"Version\": \"2012-10-17\", \"Statement\": [$stmts]}" > "/tmp/$1.json"
  mc admin policy create s3 "$1" "/tmp/$1.json"
}

user() {  # user <name> <secret> <policy>
  mc admin user add s3 "$1" "$2"
  mc admin policy attach s3 "$3" --user "$1" 2>/dev/null || true
}

# inference: reads the weights; writes predictions
policy inference-rw "models/" "predictions/"
# streaming consumer: writes window metrics and alerts
policy stream-w "" "curated/stream_windows/ monitoring/alerts/"

user inference "$INFERENCE_S3_SECRET" inference-rw
user stream "$STREAM_S3_SECRET" stream-w
echo "buckets and users ready"
