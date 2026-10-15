"""Input validation and contract errors (docs/contrato.md §2), with no heavy dependencies.

Used by the gateway (cheap checks, without decoding the image) and by the inference
service. This module does not import torch, PIL or numpy.
"""

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 4096
ALLOWED_TYPES = ("image/jpeg", "image/png")
# Actual file format (as read by PIL) that is accepted, whatever the client declares
ALLOWED_FORMATS = ("JPEG", "PNG")


class DetectionError(Exception):
    """Invalid input, carrying the contract's HTTP status (413/415/422)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class ModelNotLoaded(Exception):
    """No `champion` model is loaded (503)."""


def check_type(content_type):
    if content_type is not None and content_type not in ALLOWED_TYPES:
        raise DetectionError(415, f"unsupported content type: {content_type}")


def check_format(image_format):
    if image_format not in ALLOWED_FORMATS:
        raise DetectionError(415, f"unsupported image format: {image_format or 'unknown'}")


def check_size(size):
    if size > MAX_BYTES:
        raise DetectionError(413, f"image larger than {MAX_BYTES // (1024 * 1024)} MB")


def check_dimensions(width, height):
    if max(width, height) > MAX_SIDE:
        raise DetectionError(422, f"longest side larger than {MAX_SIDE} px")


def check_threshold(score_threshold):
    if score_threshold is not None and not 0.0 <= score_threshold <= 1.0:
        raise DetectionError(422, "score_threshold must be between 0 and 1")
