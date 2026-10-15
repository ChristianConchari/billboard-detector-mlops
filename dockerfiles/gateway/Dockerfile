# REST gateway: no torch, MLflow or boto3. Talks to the `inference` service over gRPC.
# Build context: the repository root.

# Stage 1: generate the gRPC code (grpcio-tools is not needed in the final image)
FROM python:3.12-slim AS proto
WORKDIR /build
RUN pip install --no-cache-dir grpcio-tools==1.81.1 protobuf==6.33.6
COPY proto proto
COPY scripts/gen_proto.sh scripts/gen_proto.sh
RUN mkdir -p billboard_detector && sh scripts/gen_proto.sh

# Stage 2: the final image
FROM python:3.12-slim
WORKDIR /app
COPY dockerfiles/gateway/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY dockerfiles/gateway/app.py /app/app.py
COPY billboard_detector/__init__.py billboard_detector/validation.py /app/billboard_detector/
COPY --from=proto /build/billboard_detector/grpc_gen /app/billboard_detector/grpc_gen
ENV PYTHONPATH=/app
EXPOSE 8800
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]
