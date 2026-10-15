# Consumidor de streaming: Redpanda → inference (gRPC) → ventanas por event time.
# Sin torch ni dependencias numéricas: kafka-python, gRPC y boto3. Contexto de build: raíz del repo.

FROM python:3.12-slim AS proto
WORKDIR /build
RUN pip install --no-cache-dir grpcio-tools==1.81.1 protobuf==6.33.6
COPY proto proto
COPY scripts/gen_proto.sh scripts/gen_proto.sh
RUN mkdir -p vallavision && sh scripts/gen_proto.sh

FROM python:3.12-slim
WORKDIR /app
COPY dockerfiles/stream/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY vallavision/__init__.py vallavision/streaming.py vallavision/stream_consumer.py /app/vallavision/
COPY --from=proto /build/vallavision/grpc_gen /app/vallavision/grpc_gen
ENV PYTHONPATH=/app
CMD ["python", "-m", "vallavision.stream_consumer"]
