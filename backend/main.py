import tempfile
from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.inference.predict import WasteWisePredictor


logger = logging.getLogger("wastewise.backend")
logging.basicConfig(level=logging.INFO)


# =========================================================
# Configuration
# =========================================================

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# =========================================================
# Predictor Lifecycle & Dependency Management
# =========================================================

_predictor: Optional[WasteWisePredictor] = None


def get_predictor() -> WasteWisePredictor:
    """
    Get or lazily initialize the singleton WasteWisePredictor.
    Allows worker-level reuse across requests.
    """
    global _predictor

    if _predictor is None:
        logger.info("Initializing WasteWise AI pipeline...")
        _predictor = WasteWisePredictor()
        logger.info("WasteWise AI pipeline initialized.")

    return _predictor


def set_predictor(
    custom_predictor: Optional[WasteWisePredictor],
) -> None:
    """
    Override predictor instance.
    Useful for unit testing with mocks.
    """
    global _predictor
    _predictor = custom_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Loads models once during application startup and
    performs cleanup during shutdown.
    """
    try:
        # Eagerly initialize models on startup.
        get_predictor()

    except Exception as exc:
        logger.warning(
            f"Eager model initialization deferred: {exc}"
        )

    yield

    logger.info("Shutting down WasteWise AI backend.")


# =========================================================
# Application
# =========================================================

app = FastAPI(
    title="WasteWise AI API",
    description=(
        "AI-powered waste detection, classification, "
        "OOD detection, and disposal guidance API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Health and Readiness Endpoints
# =========================================================

@app.get("/")
def root():
    return {
        "name": "WasteWise AI",
        "status": "running",
        "version": "1.0.0",
        "pipeline": [
            "YOLO11s waste detector",
            "Expanded ResNet50 classifier",
            "OOD detection",
            "Disposal guidance",
        ],
    }


@app.get("/health")
def health():
    """
    Liveness and health check endpoint.
    """
    is_ready = _predictor is not None

    device_name = (
        str(_predictor.device)
        if is_ready
        else "not_loaded"
    )

    ood_threshold = (
        _predictor.ood_detector.threshold
        if is_ready
        else None
    )

    return {
        "status": (
            "healthy"
            if is_ready
            else "initializing"
        ),
        "ready": is_ready,
        "device": device_name,
        "detector": "YOLO11s",
        "classifier": "Expanded ResNet50 Stage 2",
        "ood_detection": True,
        "ood_threshold": ood_threshold,
    }


@app.get("/ready")
def readiness():
    """
    Kubernetes/container readiness probe.

    Ensures all production models are loaded and ready
    to serve traffic.
    """
    try:
        pred = get_predictor()

        if (
            pred.detector is None
            or pred.classifier is None
            or pred.ood_detector is None
        ):
            raise RuntimeError(
                "Pipeline models not fully loaded."
            )

        return {
            "status": "ready",
            "device": str(pred.device),
            "yolo_loaded": True,
            "resnet_loaded": True,
            "ood_loaded": True,
        }

    except Exception as exc:
        logger.error(
            f"Readiness check failed: {exc}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not ready",
        )


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "YOLO detection confidence threshold"
        ),
    ),
    ood_threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "OOD similarity threshold override"
        ),
    ),
):
    # -----------------------------------------------------
    # Validate filename and extension
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image format '{extension}'. "
                f"Supported formats: "
                f"{sorted(list(ALLOWED_EXTENSIONS))}."
            ),
        )

    # -----------------------------------------------------
    # Read uploaded file
    # -----------------------------------------------------

    try:
        contents = await file.read()

    except Exception as exc:
        logger.error(
            f"Failed to read uploaded file: {exc}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    # -----------------------------------------------------
    # Validate file size
    # -----------------------------------------------------

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            ),
            detail=(
                "File size exceeds maximum limit of "
                f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
            ),
        )

    # -----------------------------------------------------
    # Validate image data using PIL
    # -----------------------------------------------------

    try:
        image_stream = io.BytesIO(contents)

        with Image.open(image_stream) as img:
            img.verify()

        # Re-open verified stream for actual validation.
        image_stream = io.BytesIO(contents)

        with Image.open(image_stream) as image:
            image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        logger.warning(
            f"Invalid image upload "
            f"({file.filename}): {exc}"
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image data.",
        )

    # -----------------------------------------------------
    # Run production AI pipeline
    # -----------------------------------------------------

    temp_path = None

    try:
        predictor = get_predictor()

        # WasteWisePredictor.predict() expects an image
        # filesystem path, not a PIL Image object.
        #
        # Materialize the validated upload as a temporary
        # image file for the production inference pipeline.

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        result = predictor.predict(
            temp_path,
            confidence_threshold=confidence_threshold,
            ood_threshold=ood_threshold,
        )

        return {
            "success": result.get(
                "success",
                True,
            ),
            "version": result.get(
                "version",
                "1.0.0",
            ),
            "filename": file.filename,
            "duration_ms": result.get(
                "duration_ms"
            ),
            "image": result.get(
                "image"
            ),
            "detector": result.get(
                "detector"
            ),
            "classifier": result.get(
                "classifier"
            ),
            "objects": result.get(
                "objects",
                [],
            ),
            "rejected_objects": result.get(
                "rejected_objects",
                [],
            ),
            "summary": result.get(
                "summary",
                {},
            ),
            "accounting": result.get(
                "accounting",
                {},
            ),
            "confidence_threshold": (
    0.35
    if confidence_threshold is None
    else float(confidence_threshold)
),
"ood_threshold": (
    0.230447
    if ood_threshold is None
    else float(ood_threshold)
),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            f"Inference failure for "
            f"{file.filename}: {exc}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Internal inference failure "
                "during image processing."
            ),
        )

    finally:
        # Always remove the temporary uploaded image.
        if temp_path:
            try:
                Path(temp_path).unlink(
                    missing_ok=True
                )

            except OSError:
                logger.warning(
                    "Failed to remove temporary "
                    f"file: {temp_path}"
                )


# =========================================================
# Run Directly
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )