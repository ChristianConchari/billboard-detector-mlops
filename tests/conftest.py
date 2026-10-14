"""Rutas de import y generación del código gRPC (no se versiona) si falta."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dockerfiles" / "gateway")]

if not (ROOT / "vallavision" / "grpc_gen" / "detector_pb2_grpc.py").exists():
    subprocess.run(["sh", str(ROOT / "scripts" / "gen_proto.sh")], check=True,
                   env={"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "PYTHON": sys.executable})
