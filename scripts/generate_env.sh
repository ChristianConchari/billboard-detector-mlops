#!/bin/sh
# Creates .env from .env.example with random secrets and AIRFLOW_UID = your UID.
# It never overwrites an existing .env (delete it by hand to regenerate; note that changing
# PG_PASSWORD with existing volumes leaves Postgres with the old password → `docker compose down -v`).
set -e
cd "$(dirname "$0")/.."
if [ -e .env ]; then
  echo ".env already exists: left untouched" >&2
  exit 1
fi
python3 - <<'PY'
import base64, os, secrets
lines = []
for line in open(".env.example"):
    key, sep, value = line.rstrip("\n").partition("=")
    if sep and value == "generate":
        value = secrets.token_urlsafe(24)
    elif sep and value == "generate-fernet":
        value = base64.urlsafe_b64encode(os.urandom(32)).decode()
    elif sep and key == "AIRFLOW_UID" and hasattr(os, "getuid"):
        value = str(os.getuid())
    lines.append(f"{key}{sep}{value}" if sep else line.rstrip("\n"))
with open(".env", "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod(".env", 0o600)
PY
echo ".env generated (mode 600). The gateway API key is GATEWAY_API_KEYS."
