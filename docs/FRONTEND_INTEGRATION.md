# WasteWise-AI — Frontend & Live Camera Integration Guide

## 1. Overview & Technology Stack

The WasteWise-AI user interface is designed as a responsive, modern single-page application (SPA) providing real-time waste analysis through two workflows:
1. **Interactive File Upload**: High-resolution image analysis with bounding box overlays, disposal instructions, and diagnostic logs.
2. **Live Camera Feed**: Low-latency video stream sampling with throttled frame inference and canvas bounding box overlays.

### Technology Stack
- **Framework**: React 18 with TypeScript (`tsc -b`)
- **Build Tool**: Vite 6 (HMR & optimized production bundling)
- **Styling**: Tailwind CSS + Lucide React iconography
- **State Management**: React Hooks (`useState`, `useRef`, `useEffect`, `useCallback`)
- **HTTP Client**: Native Fetch API with structured TypeScript typing (`frontend/src/services/api.ts`)

---

## 2. Live Camera Lifecycle (`MediaStream`)

The live camera subsystem is encapsulated in `frontend/src/components/LiveCamera.tsx`.

### 2.1 Hardware Stream Acquisition
Camera access is requested using standard WebRTC `navigator.mediaDevices.getUserMedia`:

```typescript
const constraints: MediaStreamConstraints = {
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: { ideal: "environment" } // Mobile rear camera prioritization
  },
  audio: false
};

const stream = await navigator.mediaDevices.getUserMedia(constraints);
videoRef.current.srcObject = stream;
await videoRef.current.play();
```

### 2.2 Clean Resource Release
To prevent camera lockouts and memory leaks across component mounts or route switches, all active tracks must be terminated:

```typescript
const stopCamera = () => {
  if (streamRef.current) {
    streamRef.current.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;
  }
  if (videoRef.current) {
    videoRef.current.srcObject = null;
  }
};
```

---

## 3. Frame Sampling & Throttling Strategy

Real-time browser streams run at 30–60 FPS, but inference at 60 FPS would saturate both client upload bandwidth and server GPU queues.

### 3.1 500 ms Throttled Loop
Frames are sampled at 2 FPS ($500\text{ ms}$ interval) using an active loop guarded by an `isProcessing` lock:

```typescript
useEffect(() => {
  if (!isStreaming) return;

  const intervalId = setInterval(async () => {
    if (isProcessingRef.current || !videoRef.current) return;
    isProcessingRef.current = true;

    try {
      const blob = await captureVideoFrame(videoRef.current);
      if (blob) {
        const result = await predictWaste(blob, { confidenceThreshold: 0.35 });
        setDetections(result.objects);
        renderBoundingBoxes(result.objects);
      }
    } catch (err) {
      console.warn("Frame inference failed, maintaining stream:", err);
    } finally {
      isProcessingRef.current = false;
    }
  }, 500);

  return () => clearInterval(intervalId);
}, [isStreaming]);
```

### 3.2 In-Memory JPEG Blob Conversion
The active frame is drawn to an off-screen `HTMLCanvasElement` and serialized directly to a binary JPEG blob without disk I/O:

```typescript
const captureVideoFrame = (video: HTMLVideoElement): Promise<Blob | null> => {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.resolve(null);

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
  });
};
```

---

## 4. Canvas Bounding-Box Overlay Rendering

Predictions return bounding boxes in absolute coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ relative to the source video dimensions. The client normalizes these coordinates to the CSS viewport using a transparent `<canvas>` element layered directly over the `<video>` element.

### Category Color Mapping:
| Waste Category | Stroke & Fill Color | Badge Styling |
|:---------------|:--------------------|:--------------|
| **Recyclable** | `#10B981` (Emerald) | Green badge, clean icon |
| **Hazardous**  | `#EF4444` (Red)     | Red badge, warning icon |
| **Medical Waste** | `#F59E0B` (Amber) | Amber badge, biohazard icon |
| **E-Waste**    | `#8B5CF6` (Purple)  | Purple badge, battery icon |
| **General Waste** | `#6B7280` (Gray) | Slate badge, trash icon |

### Drawing Logic:
```typescript
const renderBoundingBoxes = (objects: WasteObject[], video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const scaleX = canvas.clientWidth / video.videoWidth;
  const scaleY = canvas.clientHeight / video.videoHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  objects.forEach((obj) => {
    const [x1, y1, x2, y2] = obj.bbox;
    const sx = x1 * scaleX;
    const sy = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    // Bounding Box
    ctx.strokeStyle = getColorForCategory(obj.category);
    ctx.lineWidth = 3;
    ctx.strokeRect(sx, sy, sw, sh);

    // Label Header
    const label = `${obj.class} (${obj.confidence.toFixed(0)}%)`;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(sx, sy - 24, ctx.measureText(label).width + 12, 24);
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(label, sx + 6, sy - 7);
  });
};
```

---

## 5. Resilience & Error Recovery

1. **Dropped Frames**: If an HTTP request fails (network timeout or packet loss), the `isProcessingRef` lock is released immediately in the `finally` block, allowing subsequent frames to proceed smoothly without locking the UI.
2. **Device Permission Denied**: If the user denies camera permissions, the component renders a friendly notification instructing the user how to grant webcam permissions in browser settings.
3. **Empty or Degenerate Frames**: If the video stream is paused or zero-sized, frame capture returns `null` and skips HTTP invocation.
4. **Fallback Mode**: If camera hardware is unavailable or unsupported on the browser, the interface defaults back to the drag-and-drop file upload interface.
