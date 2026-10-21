# WasteWise-AI — REST API Contract & Specification

**Base URL**: `http://127.0.0.1:8000`  
**API Version**: `1.0.0`  
**Protocol**: HTTP/1.1  
**Content-Type**: `application/json` (responses), `multipart/form-data` (predictions)

---

## 1. Global Constraints & Limitations

- **Max Payload Size**: 25 MB (`26,214,400 bytes`). Requests exceeding this limit return `413 Payload Too Large`.
- **Supported Formats**: JPEG (`.jpg`, `.jpeg`), PNG (`.png`), WebP (`.webp`). Other formats return `400 Bad Request`.
- **CORS**: Enabled for all origins (`*`) supporting standard web frontend integration.
- **Model Concurrency**: Singleton model inference instance in FastAPI lifespan; GPU calls are executed under `@torch.no_grad()`.

---

## 2. Endpoints Overview

| Method | Endpoint   | Description | Authentication |
|:-------|:-----------|:------------|:---------------|
| `GET`  | `/`        | Service metadata and pipeline description | None |
| `GET`  | `/health`  | Liveness probe and model initialization status | None |
| `GET`  | `/ready`   | Kubernetes/container readiness probe | None |
| `POST` | `/predict` | Waste detection, classification & disposal guidance | None |

---

## 3. Endpoint Specifications

### 3.1 GET `/` (Root Information)
Returns API metadata and the active pipeline stages.

#### Response (200 OK)
```json
{
  "name": "WasteWise AI",
  "status": "running",
  "version": "1.0.0",
  "pipeline": [
    "YOLO11s waste detector",
    "Expanded ResNet50 classifier",
    "OOD detection",
    "Disposal guidance"
  ]
}
```

---

### 3.2 GET `/health` (Liveness Probe)
Checks if the web server is responsive and indicates whether model weights are currently resident in memory.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "ready": true,
  "device": "cuda",
  "detector": "YOLO11s",
  "classifier": "Expanded ResNet50 Stage 2",
  "ood_detection": true,
  "ood_threshold": 0.230447
}
```

---

### 3.3 GET `/ready` (Readiness Probe)
Returns `200 OK` when all model weights (YOLO, ResNet, OOD centroids) are fully loaded and ready to serve inference. If models are still loading or unavailable, returns `503 Service Unavailable`.

#### Response (200 OK)
```json
{
  "status": "ready",
  "device": "cuda",
  "yolo_loaded": true,
  "resnet_loaded": true,
  "ood_loaded": true
}
```

#### Response (503 Service Unavailable)
```json
{
  "detail": "Pipeline not ready"
}
```

---

### 3.4 POST `/predict` (Inference Pipeline)
Runs the two-stage waste detection, classification, OOD verification, and disposal recommendation pipeline on an uploaded image.

#### Query Parameters
| Parameter | Type | Required | Default | Description |
|:----------|:-----|:---------|:--------|:------------|
| `confidence_threshold` | `float` (0.0 to 1.0) | No | `0.35` | Override for YOLO detection confidence filter |
| `ood_threshold` | `float` (0.0 to 1.0) | No | `0.230447` | Override for OOD cosine similarity threshold |

#### Request Body
- Content-Type: `multipart/form-data`
- Field: `file` (Binary image stream)

#### Response Schema (200 OK)
```json
{
  "success": true,
  "version": "1.0.0",
  "filename": "sample_bottle.jpg",
  "duration_ms": 42.15,
  "image": {
    "width": 1280,
    "height": 720
  },
  "detector": {
    "model": "YOLO11s Balanced",
    "threshold": 0.35,
    "classes": 6
  },
  "classifier": {
    "model": "Expanded ResNet50 Stage 2",
    "classes": 19,
    "ood_enabled": true,
    "ood_threshold": 0.230447
  },
  "accounting": {
    "total_raw_detections": 1,
    "accepted_detections": 1,
    "ood_rejected": 0,
    "compatibility_rejected": 0,
    "other_rejected": 0
  },
  "summary": {
    "total_objects": 1,
    "recyclable": 1,
    "general_waste": 0,
    "hazardous": 0,
    "medical_waste": 0,
    "e_waste": 0,
    "unsupported_or_rejected": 0,
    "categories": {
      "Recyclable": 1
    }
  },
  "objects": [
    {
      "box_index": 0,
      "class": "plastic",
      "category": "Recyclable",
      "recommendation": "Rinse clean and place in the recyclable container.",
      "confidence": 98.42,
      "confidence_level": "high",
      "detector_class": "plastic",
      "detector_confidence": 88.5,
      "ood_similarity": 0.7412,
      "ood_threshold": 0.2305,
      "bbox": [150, 200, 480, 640],
      "crop_size": [330, 440],
      "is_supported": true,
      "top_predictions": [
        {"class": "plastic", "confidence": 98.42},
        {"class": "cardboard", "confidence": 1.10},
        {"class": "metal", "confidence": 0.35}
      ]
    }
  ],
  "rejected_objects": []
}
```

#### Rejected Object Schema (When Gated Out)
When a detection fails OOD similarity or detector-classifier taxonomy compatibility, it is appended to `rejected_objects`:
```json
{
  "box_index": 1,
  "rejection_reason": "compatibility_mismatch",
  "detector_class": "paper_cardboard",
  "detector_confidence": 42.1,
  "classifier_class": "battery",
  "classifier_confidence": 85.3,
  "ood_similarity": 0.1742,
  "ood_threshold": 0.2305,
  "bbox": [500, 300, 600, 420]
}
```

---

## 4. Error Responses

| HTTP Status | Error Reason | Example Response Body |
|:------------|:-------------|:----------------------|
| `400 Bad Request` | Unsupported file extension | `{"detail": "Unsupported image format '.bmp'. Supported formats: ['.jpeg', '.jpg', '.png', '.webp']."}` |
| `400 Bad Request` | Empty file uploaded | `{"detail": "Uploaded file is empty (0 bytes)."}` |
| `400 Bad Request` | Corrupt or invalid image data | `{"detail": "Corrupted or invalid image data."}` |
| `413 Payload Too Large` | Exceeded 25MB file size limit | `{"detail": "File size exceeds maximum limit of 25MB."}` |
| `500 Internal Server Error` | Pipeline exception during inference | `{"detail": "Internal inference failure during image processing."}` |
| `503 Service Unavailable` | Pipeline models not initialized | `{"detail": "Pipeline not ready"}` |
