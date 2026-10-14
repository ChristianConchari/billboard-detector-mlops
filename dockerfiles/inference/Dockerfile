# Servicio de inferencia: único dueño del modelo (gRPC). Contexto de build: raíz del repo,
# porque comparte el paquete vallavision/ con Airflow y con el gateway.
FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/inference/requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /requirements.txt
# vallavision.detector importa mlflow.pyfunc, que necesita pandas (va aparte para no
# invalidar la capa de torch)
RUN pip install --no-cache-dir pandas

COPY proto /app/proto
COPY scripts/gen_proto.sh /app/scripts/gen_proto.sh
COPY vallavision /app/vallavision
RUN rm -rf /app/vallavision/grpc_gen && sh /app/scripts/gen_proto.sh

ENV PYTHONPATH=/app
EXPOSE 50051
CMD ["python", "-m", "vallavision.grpc_server"]
