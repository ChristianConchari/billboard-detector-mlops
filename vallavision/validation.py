"""Validación y errores del contrato (docs/contrato.md §2), sin dependencias pesadas.

Lo usan el gateway (chequeos baratos, sin decodificar la imagen) y el servicio de
inferencia. Este módulo no importa torch, PIL ni numpy.
"""

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 4096
ALLOWED_TYPES = ("image/jpeg", "image/png")


class DetectionError(Exception):
    """Error de entrada con el código HTTP del contrato (413/415/422)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class ModelNotLoaded(Exception):
    """No hay modelo `champion` cargado (503)."""


def check_type(content_type):
    if content_type is not None and content_type not in ALLOWED_TYPES:
        raise DetectionError(415, f"tipo no soportado: {content_type}")


def check_size(size):
    if size > MAX_BYTES:
        raise DetectionError(413, f"imagen > {MAX_BYTES // (1024 * 1024)} MB")


def check_threshold(score_threshold):
    if score_threshold is not None and not 0.0 <= score_threshold <= 1.0:
        raise DetectionError(422, "score_threshold debe estar entre 0 y 1")
