import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  Camera,
  CameraOff,
  CheckCircle2,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";

import {
  predictWaste,
  type PredictionResponse,
  type WasteObject,
} from "../services/api";

interface LiveCameraProps {
  onClose: () => void;
}

const INFERENCE_INTERVAL_MS = 500;

/**
 * Safely format confidence values.
 *
 * Supports:
 *   0.9984 -> 99.8%
 *   99.84  -> 99.8%
 */
const formatConfidence = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0.0%";
  }

  const normalized = value > 1 ? value / 100 : value;

  const clamped = Math.max(
    0,
    Math.min(1, normalized),
  );

  return `${(clamped * 100).toFixed(1)}%`;
};

function LiveCamera({
  onClose,
}: LiveCameraProps) {
  const videoRef =
    useRef<HTMLVideoElement>(null);

  const canvasRef =
    useRef<HTMLCanvasElement>(null);

  const streamRef =
    useRef<MediaStream | null>(null);

  const inferenceBusyRef =
    useRef(false);

  const lastInferenceRef =
    useRef(0);

  const animationFrameRef =
    useRef<number | null>(null);

  const mountedRef =
    useRef(true);

  const [cameraError, setCameraError] =
    useState<string | null>(null);

  const [isCameraReady, setIsCameraReady] =
    useState(false);

  const [isInferring, setIsInferring] =
    useState(false);

  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(
      null,
    );

  const [facingMode, setFacingMode] =
    useState<"user" | "environment">(
      "environment",
    );

  /*
   * -------------------------------------------------------
   * STOP CAMERA
   * -------------------------------------------------------
   */
  const stopCamera = useCallback(() => {
    if (
      animationFrameRef.current !== null
    ) {
      cancelAnimationFrame(
        animationFrameRef.current,
      );

      animationFrameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    const video = videoRef.current;

    if (video) {
      video.pause();
      video.srcObject = null;
    }

    inferenceBusyRef.current = false;

    if (mountedRef.current) {
      setIsCameraReady(false);
      setIsInferring(false);
    }
  }, []);

  /*
   * -------------------------------------------------------
   * START CAMERA
   * -------------------------------------------------------
   */
  const startCamera = useCallback(
    async () => {
      setCameraError(null);
      setIsCameraReady(false);
      setPrediction(null);

      if (
        !navigator.mediaDevices?.getUserMedia
      ) {
        setCameraError(
          "Camera access is not supported by this browser or connection.",
        );

        return;
      }

      try {
        /*
         * Stop any existing camera before
         * requesting another one.
         */
        if (streamRef.current) {
          streamRef.current
            .getTracks()
            .forEach((track) => {
              track.stop();
            });

          streamRef.current = null;
        }

        const stream =
          await navigator.mediaDevices.getUserMedia(
            {
              video: {
                facingMode,
                width: {
                  ideal: 1280,
                },
                height: {
                  ideal: 720,
                },
              },
              audio: false,
            },
          );

        /*
         * Component may have been closed while
         * browser permission dialog was open.
         */
        if (!mountedRef.current) {
          stream
            .getTracks()
            .forEach((track) => {
              track.stop();
            });

          return;
        }

        const video =
          videoRef.current;

        if (!video) {
          stream
            .getTracks()
            .forEach((track) => {
              track.stop();
            });

          return;
        }

        streamRef.current = stream;

        video.srcObject = stream;

        await video.play();

        if (!mountedRef.current) {
          stream
            .getTracks()
            .forEach((track) => {
              track.stop();
            });

          streamRef.current = null;

          return;
        }

        setIsCameraReady(true);
      } catch (error) {
        console.error(
          "Camera error:",
          error,
        );

        if (mountedRef.current) {
          setCameraError(
            "Unable to access the camera. Please allow camera permission and try again.",
          );
        }
      }
    },
    [facingMode],
  );

  /*
   * -------------------------------------------------------
   * CAPTURE CURRENT FRAME
   * -------------------------------------------------------
   *
   * The canvas is ONLY an internal processing buffer.
   * It is never displayed to the user.
   */
  const captureFrame = useCallback(
    async () => {
      const video =
        videoRef.current;

      const canvas =
        canvasRef.current;

      if (!video || !canvas) {
        return;
      }

      if (
        video.readyState <
          HTMLMediaElement.HAVE_CURRENT_DATA ||
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {
        return;
      }

      /*
       * Match canvas resolution to the
       * actual camera frame.
       */
      canvas.width =
        video.videoWidth;

      canvas.height =
        video.videoHeight;

      const context =
        canvas.getContext("2d");

      if (!context) {
        return;
      }

      /*
       * Clear previous frame.
       */
      context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height,
      );

      /*
       * Front camera:
       * mirror the captured image so that the
       * backend sees the same orientation as
       * the user sees on screen.
       *
       * Rear camera:
       * draw normally.
       */
      if (facingMode === "user") {
        context.save();

        context.translate(
          video.videoWidth,
          0,
        );

        context.scale(-1, 1);

        context.drawImage(
          video,
          0,
          0,
          video.videoWidth,
          video.videoHeight,
        );

        context.restore();
      } else {
        context.drawImage(
          video,
          0,
          0,
          video.videoWidth,
          video.videoHeight,
        );
      }

      /*
       * Convert frame to JPEG.
       */
      const blob =
        await new Promise<Blob | null>(
          (resolve) => {
            canvas.toBlob(
              resolve,
              "image/jpeg",
              0.82,
            );
          },
        );

      if (
        !blob ||
        !mountedRef.current
      ) {
        return;
      }

      const frameFile =
        new File(
          [blob],
          `wastewise-live-${Date.now()}.jpg`,
          {
            type: "image/jpeg",
          },
        );

      inferenceBusyRef.current = true;

      if (mountedRef.current) {
        setIsInferring(true);
      }

      try {
        /*
         * Existing WasteWise backend:
         *
         * Camera frame
         *      ↓
         * YOLO11s detector
         *      ↓
         * crop
         *      ↓
         * ResNet50 classifier
         *      ↓
         * OOD
         *      ↓
         * compatibility gate
         */
        const result =
          await predictWaste(
            frameFile,
          );

        console.log(
          "========== LIVE AI RESULT ==========",
        );

        console.log(
          JSON.stringify(
            result,
            null,
            2,
          ),
        );

        console.log(
          "====================================",
        );

        if (mountedRef.current) {
          setPrediction(result);
        }
      } catch (error) {
        console.error(
          "Live inference error:",
          error,
        );
      } finally {
        inferenceBusyRef.current =
          false;

        if (mountedRef.current) {
          setIsInferring(false);
        }
      }
    },
    [facingMode],
  );

  /*
   * -------------------------------------------------------
   * CAMERA LIFECYCLE
   * -------------------------------------------------------
   */
  useEffect(() => {
    mountedRef.current = true;

    const timer =
      window.setTimeout(() => {
        if (mountedRef.current) {
          void startCamera();
        }
      }, 0);

    return () => {
      window.clearTimeout(timer);

      mountedRef.current = false;

      if (
        animationFrameRef.current !== null
      ) {
        cancelAnimationFrame(
          animationFrameRef.current,
        );

        animationFrameRef.current =
          null;
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });

        streamRef.current = null;
      }

      const video =
        videoRef.current;

      if (video) {
        video.pause();
        video.srcObject = null;
      }
    };
  }, [startCamera]);

  /*
   * -------------------------------------------------------
   * LIVE INFERENCE LOOP
   * -------------------------------------------------------
   *
   * requestAnimationFrame keeps the UI smooth.
   *
   * Actual AI inference is limited to approximately
   * one request every 500 ms.
   */
  useEffect(() => {
    if (!isCameraReady) {
      return;
    }

    lastInferenceRef.current = 0;

    let active = true;

    const tick = (
      timestamp: number,
    ) => {
      if (
        !active ||
        !mountedRef.current
      ) {
        return;
      }

      if (
        !inferenceBusyRef.current &&
        timestamp -
          lastInferenceRef.current >=
          INFERENCE_INTERVAL_MS
      ) {
        lastInferenceRef.current =
          timestamp;

        void captureFrame();
      }

      animationFrameRef.current =
        requestAnimationFrame(tick);
    };

    animationFrameRef.current =
      requestAnimationFrame(tick);

    return () => {
      active = false;

      if (
        animationFrameRef.current !== null
      ) {
        cancelAnimationFrame(
          animationFrameRef.current,
        );

        animationFrameRef.current =
          null;
      }
    };
  }, [
    isCameraReady,
    captureFrame,
  ]);

  /*
   * -------------------------------------------------------
   * SWITCH CAMERA
   * -------------------------------------------------------
   */
  const switchCamera = () => {
    stopCamera();

    setFacingMode(
      (current) =>
        current === "environment"
          ? "user"
          : "environment",
    );
  };

  /*
   * -------------------------------------------------------
   * CLOSE CAMERA
   * -------------------------------------------------------
   */
  const handleClose = () => {
    stopCamera();
    onClose();
  };

  const objects =
    prediction?.objects ?? [];

  return (
    <div className="ww-camera-overlay">
      <div className="ww-camera-panel">

        {/* =================================================
            HEADER
            ================================================= */}
        <div className="ww-camera-header">
          <div>
            <div className="ww-camera-title">
              <Camera size={19} />

              Live Waste Detection
            </div>

            <div className="ww-camera-subtitle">
              YOLO11s + ResNet50
            </div>
          </div>

          <button
            type="button"
            className="ww-camera-close"
            onClick={handleClose}
            aria-label="Close camera"
          >
            <X size={20} />
          </button>
        </div>

        {/* =================================================
            CAMERA STAGE
            ================================================= */}
        <div className="ww-camera-stage">

          {/*
            IMPORTANT:
            Only the video is visible.

            Front camera:
            CSS mirrors it.

            Rear camera:
            CSS leaves it normal.
          */}
          <video
            ref={videoRef}
            className={`ww-camera-video ${
              facingMode === "user"
                ? "ww-camera-video--mirrored"
                : ""
            }`}
            autoPlay
            muted
            playsInline
          />

          {/*
            INTERNAL PROCESSING CANVAS

            This must NEVER be visible.
            CSS hides it completely.
          */}
          <canvas
            ref={canvasRef}
            className="ww-camera-canvas"
            aria-hidden="true"
          />

          {/* =================================================
              STARTING CAMERA
              ================================================= */}
          {!isCameraReady &&
            !cameraError && (
              <div className="ww-camera-status">
                <Loader2
                  size={30}
                  className="ww-spin"
                />

                <span>
                  Starting camera...
                </span>
              </div>
            )}

          {/* =================================================
              CAMERA ERROR
              ================================================= */}
          {cameraError && (
            <div className="ww-camera-status ww-camera-error">
              <CameraOff size={34} />

              <strong>
                Camera unavailable
              </strong>

              <span>
                {cameraError}
              </span>

              <button
                type="button"
                className="ww-camera-retry"
                onClick={() =>
                  void startCamera()
                }
              >
                <RefreshCw size={16} />

                Try Again
              </button>
            </div>
          )}

          {/* =================================================
              LIVE DETECTION UI
              ================================================= */}
          {isCameraReady && (
            <>
              {/* LIVE badge */}
              <div className="ww-live-badge">
                <span className="ww-live-dot" />

                LIVE
              </div>

              {/* AI status */}
              <div className="ww-inference-badge">
                {isInferring ? (
                  <>
                    <Loader2
                      size={13}
                      className="ww-spin"
                    />

                    AI analyzing
                  </>
                ) : (
                  <>
                    <CheckCircle2
                      size={13}
                    />

                    AI ready
                  </>
                )}
              </div>

              {/* =================================================
                  DETECTION BOXES
                  ================================================= */}
              {objects.map(
                (
                  object: WasteObject,
                ) => {
                  const box =
                    object.bounding_box;

                  const predictionName =
                    object.classifier
                      .prediction;

                  const confidence =
                    object.classifier
                      .confidence;

                  const imageWidth =
                    prediction?.image
                      .width ?? 1;

                  const imageHeight =
                    prediction?.image
                      .height ?? 1;

                  return (
                    <div
                      key={
                        object.object_id
                      }
                      className="ww-live-box"
                      style={{
                        left: `${
                          (box.x1 /
                            imageWidth) *
                          100
                        }%`,

                        top: `${
                          (box.y1 /
                            imageHeight) *
                          100
                        }%`,

                        width: `${
                          (box.width /
                            imageWidth) *
                          100
                        }%`,

                        height: `${
                          (box.height /
                            imageHeight) *
                          100
                        }%`,
                      }}
                    >
                      <div className="ww-live-label">
                        {predictionName}{" "}
                        {formatConfidence(
                          confidence,
                        )}
                      </div>
                    </div>
                  );
                },
              )}

              {/* Object count */}
              <div className="ww-live-count">
                Objects detected:{" "}
                <strong>
                  {objects.length}
                </strong>
              </div>
            </>
          )}
        </div>

        {/* =================================================
            CONTROLS
            ================================================= */}
        <div className="ww-camera-controls">

          <button
            type="button"
            className="ww-camera-secondary"
            onClick={switchCamera}
            disabled={!isCameraReady}
          >
            <RefreshCw size={17} />

            Switch Camera
          </button>

          <button
            type="button"
            className="ww-camera-primary"
            onClick={() =>
              void captureFrame()
            }
            disabled={
              !isCameraReady ||
              isInferring
            }
          >
            {isInferring ? (
              <Loader2
                size={17}
                className="ww-spin"
              />
            ) : (
              <Camera size={17} />
            )}

            Capture & Analyze
          </button>

          <button
            type="button"
            className="ww-camera-stop"
            onClick={handleClose}
          >
            <CameraOff size={17} />

            Stop Camera
          </button>
        </div>

        {/* =================================================
            FOOTER
            ================================================= */}
        <div className="ww-camera-footer">
          Live detection runs approximately
          twice per second to keep the camera
          feed smooth.
        </div>

      </div>
    </div>
  );
}

export default LiveCamera;