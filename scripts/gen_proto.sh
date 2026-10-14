#!/bin/sh
# Genera vallavision/grpc_gen/ desde proto/detector.proto (requiere grpcio-tools).
set -e
cd "$(dirname "$0")/.."
mkdir -p vallavision/grpc_gen
python -m grpc_tools.protoc -I proto \
  --python_out=vallavision/grpc_gen --grpc_python_out=vallavision/grpc_gen proto/detector.proto
# protoc genera `import detector_pb2`; dentro del paquete tiene que ser relativo
sed -i 's/^import detector_pb2/from . import detector_pb2/' vallavision/grpc_gen/detector_pb2_grpc.py
touch vallavision/grpc_gen/__init__.py
