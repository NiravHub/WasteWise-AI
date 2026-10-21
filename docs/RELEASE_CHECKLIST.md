# WasteWise-AI — Production Release & Deployment Checklist

This document provides a pre-flight verification checklist for releasing and deploying the WasteWise-AI production service.

---

## 1. Checkpoints & Asset Verification

Ensure all required production weights and calibration files are present in their target paths:

| Asset | Path | Required Size | Status Check |
|:------|:-----|:--------------|:-------------|
| **YOLO11s Detector** | `runs/detect/results/taco_external_balanced/yolo11s_balanced/weights/best.pt` | ~19 MB | `Test-Path` passes |
| **ResNet50 Classifier** | `models/expanded_resnet50_stage2_best.pth` | ~90 MB | `Test-Path` passes |
| **OOD Calibration Centroids** | `models/expanded_ood_calibration.json` | ~300 KB | JSON valid (19 classes) |

---

## 2. Environment & Dependencies

- [x] **Python Environment**: Python 3.10+ (Current: 3.12 64-bit venv at `.venv/Scripts/python.exe`)
- [x] **PyTorch & CUDA**: `torch>=2.4.0` with CUDA runtime (validated on NVIDIA RTX GPU)
- [x] **Ultralytics**: `ultralytics>=8.3.0`
- [x] **Node.js**: Node 18+ / npm (Current: Node v20+ with Vite 6)
- [x] **FastAPI & Uvicorn**: `fastapi>=0.110.0`, `uvicorn>=0.28.0`

---

## 3. Automated Validation Test Gates

Before staging or shipping code changes, all tests must pass cleanly:

### 3.1 Backend & AI Regression Suite
Run all unit and integration tests:
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
**Expected Outcome**: 20/20 tests passing in under 5 seconds with zero failures.

### 3.2 Frontend TypeScript Compilation & Production Build
Verify TypeScript types and asset bundling:
```powershell
cd frontend
npm run build
```
**Expected Outcome**: Zero TypeScript errors (`tsc -b`), clean Vite production bundle generated in `frontend/dist/`.

---

## 4. Operational Launch Commands

### 4.1 Backend Service Launch
To run the high-performance FastAPI server with lifespan singleton model loading:
```powershell
# From project root:
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Verify readiness:
```powershell
curl http://127.0.0.1:8000/ready
# Expected: {"status":"ready","device":"cuda","yolo_loaded":true,"resnet_loaded":true,"ood_loaded":true}
```

### 4.2 Frontend Application Launch
To launch the Vite development server with HMR:
```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
Open browser at `http://127.0.0.1:5173`.

---

## 5. Pre-Flight Verification Sign-Off

- [x] **Model Accounting Invariant**: Tested and verified (`raw == accepted + rejected`).
- [x] **OOD Gate**: Tested and verified (cosine similarity threshold = 0.230447).
- [x] **Detector Threshold**: Calibrated to empirical optimum (`0.35`).
- [x] **Input Sanitization**: 25MB file size limit and in-memory PIL byte stream validation enforced.
- [x] **CORS Configuration**: Wildcard headers enabled for local and cloud web clients.
- [x] **Live Camera Subsystem**: Stream lifecycle management and 500 ms frame throttle verified.
