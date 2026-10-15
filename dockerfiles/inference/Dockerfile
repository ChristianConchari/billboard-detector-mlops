# Inference service: the only owner of the model (gRPC). Build context: the repository root,
# because it shares the billboard_detector/ package with Airflow and the gateway.
FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/inference/requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /requirements.txt
# billboard_detector.detector imports mlflow.pyfunc, which needs pandas (installed separately
# so the torch layer is not invalidated)
RUN pip install --no-cache-dir pandas==3.0.6 numpy==2.5.4

COPY proto /app/proto
COPY scripts/gen_proto.sh /app/scripts/gen_proto.sh
COPY billboard_detector /app/billboard_detector
RUN rm -rf /app/billboard_detector/grpc_gen && sh /app/scripts/gen_proto.sh

ENV PYTHONPATH=/app
EXPOSE 50051
CMD ["python", "-m", "billboard_detector.grpc_server"]
