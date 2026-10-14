# Contexto de build: raíz del repo (comparte el paquete vallavision/ con Airflow)
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

COPY dockerfiles/fastapi/requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /requirements.txt
# vallavision.detector importa mlflow.pyfunc, que necesita pandas (va aparte para no
# invalidar la capa de torch)
RUN pip install --no-cache-dir pandas

COPY dockerfiles/fastapi/app.py /app/app.py
COPY proto /app/proto
COPY scripts/gen_proto.sh /app/scripts/gen_proto.sh
COPY vallavision /app/vallavision
RUN rm -rf /app/vallavision/grpc_gen && sh /app/scripts/gen_proto.sh

ENV PYTHONPATH=/app
EXPOSE 8800 50051
