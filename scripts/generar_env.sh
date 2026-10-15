#!/bin/sh
# Crea .env a partir de .env.example con secretos aleatorios y AIRFLOW_UID = tu UID.
# No pisa un .env existente (borrarlo a mano para regenerar; ojo: cambiar PG_PASSWORD con
# volúmenes ya creados deja a Postgres con la contraseña vieja → `docker compose down -v`).
set -e
cd "$(dirname "$0")/.."
if [ -e .env ]; then
  echo ".env ya existe: no se modifica" >&2
  exit 1
fi
python3 - <<'PY'
import base64, os, secrets
lines = []
for line in open(".env.example"):
    key, sep, value = line.rstrip("\n").partition("=")
    if sep and value == "generar":
        value = secrets.token_urlsafe(24)
    elif sep and value == "generar-fernet":
        value = base64.urlsafe_b64encode(os.urandom(32)).decode()
    elif sep and key == "AIRFLOW_UID" and hasattr(os, "getuid"):
        value = str(os.getuid())
    lines.append(f"{key}{sep}{value}" if sep else line.rstrip("\n"))
with open(".env", "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod(".env", 0o600)
PY
echo ".env generado (permisos 600). La API key del gateway es GATEWAY_API_KEYS."
