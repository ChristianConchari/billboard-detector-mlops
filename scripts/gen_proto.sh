#!/bin/sh
# Generates billboard_detector/grpc_gen/ from proto/detector.proto (requires grpcio-tools).
set -e
cd "$(dirname "$0")/.."
mkdir -p billboard_detector/grpc_gen
"${PYTHON:-python}" -m grpc_tools.protoc -I proto \
  --python_out=billboard_detector/grpc_gen --grpc_python_out=billboard_detector/grpc_gen proto/detector.proto
# protoc emits `import detector_pb2`; inside the package it has to be relative
sed -i 's/^import detector_pb2/from . import detector_pb2/' billboard_detector/grpc_gen/detector_pb2_grpc.py
touch billboard_detector/grpc_gen/__init__.py
