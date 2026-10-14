# Gateway REST: sin torch, MLflow ni boto3. Habla con el servicio `inference` por gRPC.
# Contexto de build: raíz del repo.

# Etapa 1: generar el código gRPC (grpcio-tools no hace falta en la imagen final)
FROM python:3.12-slim AS proto
WORKDIR /build
RUN pip install --no-cache-dir "grpcio-tools>=1.60"
COPY proto proto
COPY scripts/gen_proto.sh scripts/gen_proto.sh
RUN mkdir -p vallavision && sh scripts/gen_proto.sh

# Etapa 2: la imagen final
FROM python:3.12-slim
WORKDIR /app
COPY dockerfiles/gateway/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY dockerfiles/gateway/app.py /app/app.py
COPY vallavision/__init__.py vallavision/validation.py /app/vallavision/
COPY --from=proto /build/vallavision/grpc_gen /app/vallavision/grpc_gen
ENV PYTHONPATH=/app
EXPOSE 8800
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]
