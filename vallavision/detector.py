"""Detector de vallas con la salida de docs/contrato.md §2.

`BillboardDetector` es PyTorch puro (lo usan el DAG para evaluar y el servicio para
servir) y `DetectorPyfunc` lo envuelve como modelo pyfunc de MLflow. Así lo que se
evalúa es exactamente lo que se registra.
"""

import io
import json
import os

import mlflow.pyfunc

# Respaldo si el checkpoint no trae calibration.json
FALLBACK_THRESHOLD = 0.5


def read_calibrated_threshold(weights_dir):
    """Umbral calibrado sobre validación (calibration.json), o None si no existe."""
    path = os.path.join(weights_dir, "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return float(json.load(f)["score_threshold"])


class BillboardDetector:
    def __init__(self, weights_dir, label_map=None):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

        # Sin esto, el backbone de timm se baja al construir el modelo aunque los pesos
        # ya vengan en el checkpoint
        config = AutoConfig.from_pretrained(weights_dir)
        if hasattr(config, "use_pretrained_backbone"):
            config.use_pretrained_backbone = False

        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(weights_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(weights_dir, config=config).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # None → las etiquetas del checkpoint pasan tal cual
        self.label_map = label_map
        self.default_threshold = read_calibrated_threshold(weights_dir) or FALLBACK_THRESHOLD

    @property
    def labels(self):
        names = self.id2label.values()
        return sorted({self.label_map.get(n) for n in names if n in self.label_map}
                      if self.label_map else set(names))

    def detect(self, image, score_threshold=None):
        """Devuelve ({width, height}, detecciones) con cajas xyxy en píxeles absolutos.

        `score_threshold=None` usa el umbral calibrado del checkpoint.
        """
        from PIL import Image

        if score_threshold is None:
            score_threshold = self.default_threshold
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        width, height = image.size

        with self._torch.inference_mode():
            outputs = self.model(**self.processor(images=image, return_tensors="pt"))
        result = self.processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=[(height, width)]
        )[0]

        detections = []
        for label_id, score, box in zip(result["labels"], result["scores"], result["boxes"]):
            label = self.id2label.get(int(label_id), "")
            if self.label_map is not None:
                label = self.label_map.get(label)
            if not label:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            detections.append({
                "label": label,
                "score": round(float(score), 4),
                "box": [
                    round(max(0.0, x0), 1), round(max(0.0, y0), 1),
                    round(min(float(width), x1), 1), round(min(float(height), y1), 1),
                ],
            })
        detections.sort(key=lambda d: d["score"], reverse=True)
        return {"width": width, "height": height}, detections


class DetectorPyfunc(mlflow.pyfunc.PythonModel):
    """Entrada: DataFrame con la columna `image` (bytes JPEG/PNG), una fila por imagen.
    Salida: una entrada por imagen con `image` y `detections` como en el contrato.
    Sin `score_threshold` se usa el umbral calibrado del checkpoint."""

    def load_context(self, context):
        label_map = (context.model_config or {}).get("label_map")
        self.detector = BillboardDetector(context.artifacts["weights"], label_map=label_map)

    def predict(self, context, model_input, params=None):
        threshold = (params or {}).get("score_threshold")
        out = []
        for image in model_input["image"]:
            size, detections = self.detector.detect(
                image, score_threshold=None if threshold is None else float(threshold))
            out.append({"image": size, "detections": detections})
        return out
