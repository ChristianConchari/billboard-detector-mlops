# Streaming consumer: Redpanda → inference (gRPC) → event-time windows.
# No torch and no numeric dependencies: kafka-python, gRPC and boto3. Build context: the repository root.

FROM python:3.12-slim AS proto
WORKDIR /build
RUN pip install --no-cache-dir grpcio-tools==1.81.1 protobuf==6.33.6
COPY proto proto
COPY scripts/gen_proto.sh scripts/gen_proto.sh
RUN mkdir -p billboard_detector && sh scripts/gen_proto.sh

FROM python:3.12-slim
WORKDIR /app
COPY dockerfiles/stream/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY billboard_detector/__init__.py billboard_detector/streaming.py billboard_detector/stream_consumer.py /app/billboard_detector/
COPY --from=proto /build/billboard_detector/grpc_gen /app/billboard_detector/grpc_gen
ENV PYTHONPATH=/app
CMD ["python", "-m", "billboard_detector.stream_consumer"]
