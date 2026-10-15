"""Import paths, plus generating the gRPC code (not versioned) if it is missing or stale."""

import os
import subprocess
import sys
from pathlib import Path

# Gateway keys for the tests (app.py refuses to start without keys)
os.environ.setdefault("GATEWAY_API_KEYS", "user-key")
os.environ.setdefault("GATEWAY_ADMIN_KEYS", "admin-key")
os.environ.setdefault("GATEWAY_RATE_LIMIT_PER_MIN", "1000")

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dockerfiles" / "gateway")]

_GEN = ROOT / "billboard_detector" / "grpc_gen" / "detector_pb2_grpc.py"
if not _GEN.exists() or _GEN.stat().st_mtime < (ROOT / "proto" / "detector.proto").stat().st_mtime:
    subprocess.run(["sh", str(ROOT / "scripts" / "gen_proto.sh")], check=True,
                   env={"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "PYTHON": sys.executable})
