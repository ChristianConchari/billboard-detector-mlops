"""API REST de VallaVision (docs/contrato.md §2)."""

import asyncio
import contextlib
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from vallavision.service import MAX_BYTES, DetectionError, DetectionService, ModelNotLoaded

RELOAD_RETRY_SECONDS = 30

service = DetectionService()


class ModelRef(BaseModel):
    name: str
    version: str
    alias: str


class ImageSize(BaseModel):
    width: int
    height: int


class Detection(BaseModel):
    label: str
    score: float
    box: list[float]  # [x_min, y_min, x_max, y_max] en píxeles de la imagen original


class DetectResponse(BaseModel):
    request_id: str
    model: ModelRef
    image: ImageSize
    detections: list[Detection]
    latency_ms: float


async def _load_until_ready():
    """El champion puede no existir todavía (el DAG no corrió): reintenta en segundo plano."""
    while True:
        try:
            version = await asyncio.to_thread(service.load_champion)
            print(f"modelo champion v{version} cargado")
        except Exception as e:  # noqa: BLE001
            if not service.loaded:
                print(f"sin champion todavía ({type(e).__name__}): {e}")
        # Con modelo cargado sigue chequeando para recoger promociones nuevas
        await asyncio.sleep(RELOAD_RETRY_SECONDS)


@contextlib.asynccontextmanager
async def lifespan(_):
    task = asyncio.create_task(_load_until_ready())
    yield
    task.cancel()


app = FastAPI(title="VallaVision", version="1", lifespan=lifespan)


@app.get("/health")
def health():
    if not service.loaded:
        raise HTTPException(503, detail={"status": "unavailable", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}


@app.get("/v1/model")
def model_info():
    if not service.loaded:
        raise HTTPException(503, detail="no hay modelo champion cargado")
    return service.info


@app.post("/v1/reload")
def reload_model():
    """Fuerza la recarga del champion (por ejemplo, justo después de promover una versión)."""
    try:
        return {"version": service.load_champion()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, detail=f"no se pudo cargar el champion: {e}") from e


@app.post("/v1/detect", response_model=DetectResponse)
def detect(
    background: BackgroundTasks,
    image: UploadFile = File(...),
    score_threshold: Optional[float] = Query(
        None, description="0.0–1.0; por defecto el umbral calibrado del modelo"),
):
    data = image.file.read(MAX_BYTES + 1)
    try:
        response = service.detect(data, content_type=image.content_type,
                                  score_threshold=score_threshold)
    except ModelNotLoaded:
        raise HTTPException(503, detail="no hay modelo champion cargado") from None
    except DetectionError as e:
        raise HTTPException(e.status, detail=e.message) from None
    background.add_task(service.persist, response)
    return response
