"""Rutas de import y generación del código gRPC (no se versiona) si falta o quedó viejo."""

import os
import subprocess
import sys
from pathlib import Path

# Claves del gateway para los tests (app.py se niega a arrancar sin claves)
os.environ.setdefault("GATEWAY_API_KEYS", "clave-usuario")
os.environ.setdefault("GATEWAY_ADMIN_KEYS", "clave-admin")
os.environ.setdefault("GATEWAY_RATE_LIMIT_PER_MIN", "1000")

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dockerfiles" / "gateway")]

_GEN = ROOT / "vallavision" / "grpc_gen" / "detector_pb2_grpc.py"
if not _GEN.exists() or _GEN.stat().st_mtime < (ROOT / "proto" / "detector.proto").stat().st_mtime:
    subprocess.run(["sh", str(ROOT / "scripts" / "gen_proto.sh")], check=True,
                   env={"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "PYTHON": sys.executable})
