import { useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  Clock3,
  Image as ImageIcon,
  Layers3,
  Loader2,
  Maximize2,
  Minus,
  MousePointer2,
  Plus,
  Recycle,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

import {
  predictWaste,
  type PredictionResponse,
  type WasteObject,
} from "../services/api";

interface WasteAnalyzerProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onClear: () => void;
}

type OverlayObject = WasteObject & {
  left: number;
  top: number;
  width: number;
  height: number;
  accepted: boolean;
};

function formatPercent(value: number) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return "0.0%";
  }

  return `${Math.max(0, Math.min(100, numericValue)).toFixed(1)}%`;
}

function humanizeLabel(value: string | null | undefined) {
  if (!value) return "Unknown";

  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function confidenceClass(confidence: number) {
  if (confidence >= 85) return "high";
  if (confidence >= 60) return "medium";
  return "low";
}

function getWasteIcon(label: string | null | undefined) {
  const value = (label ?? "").toLowerCase();

  if (
    value.includes("plastic") ||
    value.includes("glass") ||
    value.includes("metal") ||
    value.includes("paper") ||
    value.includes("cardboard") ||
    value.includes("biological") ||
    value.includes("trash") ||
    value.includes("battery")
  ) {
    return Recycle;
  }

  if (
    value.includes("medical") ||
    value.includes("gauze") ||
    value.includes("glove") ||
    value.includes("urine") ||
    value.includes("test")
  ) {
    return ShieldCheck;
  }

  return CircleHelp;
}

function rejectionLabel(reason: string | null | undefined) {
  switch (reason) {
    case "ood_rejection":
      return "OOD rejection";

    case "compatibility_mismatch":
      return "Compatibility mismatch";

    default:
      return "Pipeline rejection";
  }
}

function rejectionDescription(reason: string | null | undefined) {
  switch (reason) {
    case "ood_rejection":
      return "The classifier considered this detection outside the supported visual distribution.";

    case "compatibility_mismatch":
      return "The detector and classifier predictions were semantically incompatible.";

    default:
      return "This detection did not pass the final WasteWise verification layer.";
  }
}

interface ObjectCardProps {
  item: WasteObject;
  active: boolean;
  expanded: boolean;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
  onToggleDetails: (id: number) => void;
}

function ObjectCard({
  item,
  active,
  expanded,
  onHover,
  onSelect,
  onToggleDetails,
}: ObjectCardProps) {
  const accepted = item.accepted !== false;
  const prediction = item.classifier?.prediction ?? null;
  const confidence = item.classifier?.confidence ?? 0;
  const Icon = getWasteIcon(prediction);

  if (!accepted) {
    return (
      <motion.article
        layout
        className={`ww-object-card ww-object-card--rejected ${
          active ? "is-active" : ""
        }`}
        onMouseEnter={() => onHover(item.object_id)}
        onMouseLeave={() => onHover(null)}
        onClick={() => onSelect(item.object_id)}
        data-object-id={item.object_id}
        whileHover={{ y: -2 }}
        transition={{ duration: 0.18 }}
      >
        <div className="ww-object-card-header">
          <div className="ww-object-title">
            <div className="ww-object-icon ww-object-icon--rejected">
              <XCircle size={17} />
            </div>

            <div>
              <span className="ww-object-number">
                Object {String(item.object_id).padStart(2, "0")}
              </span>

              <h4>{rejectionLabel(item.rejection_reason)}</h4>
            </div>
          </div>

          <div className="ww-rejection-badge">FILTERED</div>
        </div>

        <div className="ww-rejection-summary">
          <span>
            {humanizeLabel(item.detector?.class)}
            <b>{formatPercent(item.detector?.confidence ?? 0)}</b>
          </span>

          <span className="ww-rejection-arrow">→</span>

          <span>
            {prediction ? humanizeLabel(prediction) : "Unsupported"}
            <b>{formatPercent(confidence)}</b>
          </span>
        </div>

        <div className="ww-rejection-reason">
          <div className="ww-rejection-reason-title">
            <AlertCircle size={14} />
            <strong>{rejectionLabel(item.rejection_reason)}</strong>
          </div>

          <p>{rejectionDescription(item.rejection_reason)}</p>
        </div>

        <div className="ww-object-mini-grid">
          <div>
            <span>OOD</span>
            <strong>
              {item.classifier?.is_supported ? "Supported" : "Rejected"}
            </strong>
          </div>

          <div>
            <span>Similarity</span>
            <strong>
              {Number(item.classifier?.ood_similarity ?? 0).toFixed(3)}
            </strong>
          </div>

          <div>
            <span>Decision</span>
            <strong className="ww-unsupported">Rejected</strong>
          </div>
        </div>

        <button
          type="button"
          className="ww-card-details-button"
          onClick={(event) => {
            event.stopPropagation();
            onToggleDetails(item.object_id);
          }}
        >
          <span>Technical details</span>
          {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>

        <AnimatePresence initial={false}>
          {expanded && (
            <motion.div
              className="ww-expanded-details"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <div>
                <span>Detector</span>
                <strong>
                  {humanizeLabel(item.detector?.class)}{" "}
                  {formatPercent(item.detector?.confidence ?? 0)}
                </strong>
              </div>

              <div>
                <span>Classifier</span>
                <strong>
                  {prediction
                    ? humanizeLabel(prediction)
                    : "Unsupported"}{" "}
                  {formatPercent(confidence)}
                </strong>
              </div>

              <div>
                <span>OOD threshold</span>
                <strong>
                  {Number(
                    item.classifier?.ood_threshold ?? 0,
                  ).toFixed(4)}
                </strong>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.article>
    );
  }

  return (
    <motion.article
      layout
      className={`ww-object-card ww-object-card--accepted ${
        active ? "is-active" : ""
      }`}
      onMouseEnter={() => onHover(item.object_id)}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect(item.object_id)}
      data-object-id={item.object_id}
      whileHover={{ y: -2 }}
      transition={{ duration: 0.18 }}
    >
      <div className="ww-object-card-header">
        <div className="ww-object-title">
          <div className="ww-object-icon">
            <Icon size={17} />
          </div>

          <div>
            <span className="ww-object-number">
              Object {String(item.object_id).padStart(2, "0")}
            </span>

            <h4>{humanizeLabel(prediction)}</h4>
          </div>
        </div>

        <div
          className={`ww-confidence ${confidenceClass(confidence)}`}
        >
          {formatPercent(confidence)}
        </div>
      </div>

      <div className="ww-object-status">
        <CheckCircle2 size={13} />
        VERIFIED BY WASTEWISE
      </div>

      <div className="ww-primary-object-stats">
        <div>
          <span>Detector</span>
          <strong>
            {formatPercent(item.detector?.confidence ?? 0)}
          </strong>
        </div>

        <div>
          <span>Classifier</span>
          <strong>{formatPercent(confidence)}</strong>
        </div>

        <div>
          <span>OOD</span>
          <strong className="ww-supported">
            <Check size={13} />
            Supported
          </strong>
        </div>
      </div>

      <div className="ww-category-line">
        <span>Waste stream</span>
        <strong>{item.classifier?.category ?? "Unknown"}</strong>
      </div>

      <button
        type="button"
        className="ww-card-details-button"
        onClick={(event) => {
          event.stopPropagation();
          onToggleDetails(item.object_id);
        }}
      >
        <span>Technical details</span>
        {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            className="ww-expanded-details"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <div>
              <span>Detector class</span>
              <strong>
                {humanizeLabel(item.detector?.class)}
              </strong>
            </div>

            <div>
              <span>Detector confidence</span>
              <strong>
                {formatPercent(item.detector?.confidence ?? 0)}
              </strong>
            </div>

            <div>
              <span>Classifier confidence</span>
              <strong>{formatPercent(confidence)}</strong>
            </div>

            <div>
              <span>Confidence level</span>
              <strong>
                {humanizeLabel(
                  item.classifier?.confidence_level,
                )}
              </strong>
            </div>

            <div>
              <span>OOD similarity</span>
              <strong>
                {Number(
                  item.classifier?.ood_similarity ?? 0,
                ).toFixed(3)}
              </strong>
            </div>

            <div>
              <span>OOD threshold</span>
              <strong>
                {Number(
                  item.classifier?.ood_threshold ?? 0,
                ).toFixed(4)}
              </strong>
            </div>

            {item.classifier?.recommendation && (
              <div className="ww-detail-recommendation">
                <Recycle size={14} />
                <span>
                  {item.classifier.recommendation}
                </span>
              </div>
            )}

            {item.classifier?.top_predictions?.length > 0 && (
              <div className="ww-detail-alternatives">
                <span>Alternative predictions</span>

                <div>
                  {item.classifier.top_predictions
                    .slice(0, 3)
                    .map((predictionItem) => (
                      <span
                        key={`${item.object_id}-${predictionItem.class}`}
                      >
                        {humanizeLabel(predictionItem.class)}
                        <b>
                          {formatPercent(
                            predictionItem.confidence,
                          )}
                        </b>
                      </span>
                    ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}

export default function WasteAnalyzer({
  selectedFile,
  previewUrl,
  onClear,
}: WasteAnalyzerProps) {
  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [hoveredObjectId, setHoveredObjectId] =
    useState<number | null>(null);

  const [selectedObjectId, setSelectedObjectId] =
    useState<number | null>(null);

  const [expandedObjectId, setExpandedObjectId] =
    useState<number | null>(null);

  const [zoom, setZoom] = useState(1);

  const imageViewportRef = useRef<HTMLDivElement | null>(null);
  const requestIdRef = useRef(0);

  const activePrediction =
    prediction &&
    selectedFile &&
    prediction.filename === selectedFile.name
      ? prediction
      : null;

  const analyzeWaste = async () => {
    if (!selectedFile) return;

    const fileBeingAnalyzed = selectedFile;
    const requestId = ++requestIdRef.current;

    setIsAnalyzing(true);
    setError(null);
    setPrediction(null);
    setHoveredObjectId(null);
    setSelectedObjectId(null);
    setExpandedObjectId(null);
    setZoom(1);

    try {
      const result = await predictWaste(fileBeingAnalyzed);

      if (requestId !== requestIdRef.current) {
        return;
      }

      setPrediction(result);
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the image.",
      );
    } finally {
      if (requestId === requestIdRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  const acceptedObjects = activePrediction?.objects ?? [];

  const rejectedObjects =
    activePrediction?.rejected_objects ?? [];

  const rawDetections =
    activePrediction?.summary.raw_detections ??
    activePrediction?.detector.raw_detections ??
    activePrediction?.detector.detections ??
    acceptedObjects.length + rejectedObjects.length;

  const acceptedDetections =
    activePrediction?.summary.accepted_objects ??
    acceptedObjects.length;

  const totalRejected =
    activePrediction?.summary.total_rejections ??
    rejectedObjects.length;

  const oodRejected =
    activePrediction?.summary.ood_rejections ??
    rejectedObjects.filter(
      (item) => item.rejection_reason === "ood_rejection",
    ).length;

  const compatibilityRejected =
    activePrediction?.summary.compatibility_rejections ??
    rejectedObjects.filter(
      (item) =>
        item.rejection_reason ===
        "compatibility_mismatch",
    ).length;

  const overlayObjects = useMemo<OverlayObject[]>(() => {
    if (!activePrediction) return [];

    const accepted = activePrediction.objects ?? [];
    const rejected =
      activePrediction.rejected_objects ?? [];

    const combined = [
      ...accepted.map((item) => ({
        ...item,
        accepted: true,
      })),
      ...rejected.map((item) => ({
        ...item,
        accepted: false,
      })),
    ];

    return combined.map((item) => {
      const box = item.bounding_box;

      return {
        ...item,
        left:
          (box.x1 / activePrediction.image.width) * 100,
        top:
          (box.y1 / activePrediction.image.height) * 100,
        width:
          ((box.x2 - box.x1) /
            activePrediction.image.width) *
          100,
        height:
          ((box.y2 - box.y1) /
            activePrediction.image.height) *
          100,
      };
    });
  }, [activePrediction]);

  const selectedObject =
    overlayObjects.find(
      (item) => item.object_id === selectedObjectId,
    ) ?? null;

  const selectObject = (id: number) => {
    setSelectedObjectId((current) =>
      current === id ? null : id,
    );

    setHoveredObjectId(id);
  };

  const clearSelection = () => {
    setSelectedObjectId(null);
    setHoveredObjectId(null);
  };

  const zoomIn = () => {
    setZoom((current) =>
      Math.min(2.5, Number((current + 0.15).toFixed(2))),
    );
  };

  const zoomOut = () => {
    setZoom((current) =>
      Math.max(0.7, Number((current - 0.15).toFixed(2))),
    );
  };

  const resetZoom = () => {
    setZoom(1);
  };

  const toggleFullscreen = async () => {
    const element = imageViewportRef.current;

    if (!element) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await element.requestFullscreen();
      }
    } catch {
      // Fullscreen is an enhancement; ignore browser restrictions.
    }
  };

  const clearAnalysis = () => {
    requestIdRef.current += 1;

    setPrediction(null);
    setError(null);
    setIsAnalyzing(false);
    setHoveredObjectId(null);
    setSelectedObjectId(null);
    setExpandedObjectId(null);
    setZoom(1);

    onClear();
  };

  const analyzeAnotherImage = () => {
    requestIdRef.current += 1;

    setPrediction(null);
    setError(null);
    setIsAnalyzing(false);
    setHoveredObjectId(null);
    setSelectedObjectId(null);
    setExpandedObjectId(null);
    setZoom(1);
  };

  return (
    <section className="ww-analyzer">
      {!selectedFile && (
        <motion.div
          className="ww-empty-state"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="ww-empty-icon">
            <ScanSearch size={30} />
          </div>

          <div className="ww-empty-copy">
            <span className="ww-empty-eyebrow">
              READY STATE
            </span>

            <h3>
              Your waste intelligence console is ready.
            </h3>

            <p>
              Upload an image and WasteWise will detect,
              classify, verify, and explain each candidate
              object.
            </p>
          </div>

          <div className="ww-empty-pipeline">
            <span>
              <ScanSearch size={14} />
              Detect
            </span>

            <span>
              <Sparkles size={14} />
              Classify
            </span>

            <span>
              <ShieldCheck size={14} />
              Verify
            </span>
          </div>
        </motion.div>
      )}

      {selectedFile && (
        <motion.div
          className="ww-analysis-workspace"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
        >
          {/* =====================================================
              TOP ANALYSIS HEADER
          ===================================================== */}

          <div className="ww-analysis-topbar">
            <div>
              <div className="ww-analysis-kicker">
                <span className="ww-live-dot" />
                {activePrediction
                  ? "PERCEPTION COMPLETE"
                  : "VISION INPUT"}
              </div>

              <h2>
                {activePrediction
                  ? "Waste intelligence."
                  : "Ready for perception."}
              </h2>

              <p title={selectedFile.name}>
                {selectedFile.name}
              </p>
            </div>

            <div className="ww-analysis-top-actions">
              {activePrediction && (
                <div className="ww-analysis-pill">
                  <CheckCircle2 size={14} />
                  {acceptedDetections} trusted
                </div>
              )}

              <button
                type="button"
                className="ww-icon-button"
                onClick={clearAnalysis}
                title="Remove image"
                aria-label="Remove image"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* =====================================================
              MAIN VISION WORKSPACE
          ===================================================== */}

          <div className="ww-analysis-grid">
            {/* ===================================================
                IMAGE / VISION CANVAS
            =================================================== */}

            <div className="ww-preview-panel ww-vision-panel">
              <div className="ww-panel-heading">
                <div>
                  <span>VISION CANVAS</span>

                  <h3>
                    {activePrediction
                      ? `${rawDetections} detection ${
                          rawDetections === 1
                            ? "proposal"
                            : "proposals"
                        }`
                      : "Image stream"}
                  </h3>
                </div>

                {activePrediction && (
                  <div className="ww-resolution">
                    {activePrediction.image.width} ×{" "}
                    {activePrediction.image.height}
                  </div>
                )}
              </div>

              <div className="ww-image-toolbar">
                <div className="ww-image-toolbar-left">
                  <span>
                    <Layers3 size={14} />
                    {activePrediction
                      ? "LIVE PERCEPTION"
                      : "SOURCE IMAGE"}
                  </span>
                </div>

                <div className="ww-image-controls">
                  <button
                    type="button"
                    onClick={zoomOut}
                    disabled={zoom <= 0.7}
                    aria-label="Zoom out"
                  >
                    <Minus size={15} />
                  </button>

                  <button
                    type="button"
                    className="ww-zoom-value"
                    onClick={resetZoom}
                    title="Reset zoom"
                  >
                    {Math.round(zoom * 100)}%
                  </button>

                  <button
                    type="button"
                    onClick={zoomIn}
                    disabled={zoom >= 2.5}
                    aria-label="Zoom in"
                  >
                    <Plus size={15} />
                  </button>

                  <button
                    type="button"
                    onClick={resetZoom}
                    aria-label="Reset zoom"
                    title="Reset zoom"
                  >
                    <RotateCcw size={14} />
                  </button>

                  <button
                    type="button"
                    onClick={toggleFullscreen}
                    aria-label="Fullscreen"
                    title="Fullscreen"
                  >
                    <Maximize2 size={15} />
                  </button>
                </div>
              </div>

              <div
                className="ww-image-stage"
                ref={imageViewportRef}
              >
                {previewUrl ? (
                  <div
                    className="ww-image-transform"
                    style={{
                      transform: `scale(${zoom})`,
                    }}
                  >
                    <div className="ww-image-wrapper">
                      <img
                        src={previewUrl}
                        alt="Waste selected for analysis"
                      />

                      {activePrediction &&
                        overlayObjects.map((item) => {
                          const isActive =
                            hoveredObjectId ===
                              item.object_id ||
                            selectedObjectId ===
                              item.object_id;

                          const isDimmed =
                            (hoveredObjectId !== null ||
                              selectedObjectId !== null) &&
                            !isActive;

                          const predictionLabel =
                            item.classifier?.prediction;

                          const label = item.accepted
                            ? `${humanizeLabel(
                                predictionLabel,
                              )} ${formatPercent(
                                item.classifier
                                  ?.confidence ?? 0,
                              )}`
                            : `FILTERED · OBJECT ${item.object_id}`;

                          return (
                            <button
                              type="button"
                              key={`overlay-${item.object_id}`}
                              className={[
                                "ww-detection-box",
                                item.accepted
                                  ? "ww-detection-box--accepted"
                                  : "ww-detection-box--rejected",
                                isActive
                                  ? "is-active"
                                  : "",
                                isDimmed
                                  ? "is-dimmed"
                                  : "",
                              ]
                                .filter(Boolean)
                                .join(" ")}
                              style={{
                                left: `${item.left}%`,
                                top: `${item.top}%`,
                                width: `${item.width}%`,
                                height: `${item.height}%`,
                              }}
                              onMouseEnter={() =>
                                setHoveredObjectId(
                                  item.object_id,
                                )
                              }
                              onMouseLeave={() =>
                                setHoveredObjectId(null)
                              }
                              onClick={() =>
                                selectObject(
                                  item.object_id,
                                )
                              }
                            >
                              <span>
                                <b>
                                  {String(
                                    item.object_id,
                                  ).padStart(2, "0")}
                                </b>

                                {label}
                              </span>
                            </button>
                          );
                        })}

                      {activePrediction &&
                        overlayObjects.length > 0 && (
                          <div className="ww-image-scanline" />
                        )}
                    </div>
                  </div>
                ) : (
                  <div className="ww-no-preview">
                    <Upload size={28} />
                    <span>Preview unavailable</span>
                  </div>
                )}

                {activePrediction &&
                  selectedObject && (
                    <motion.div
                      className="ww-selected-object-hud"
                      initial={{
                        opacity: 0,
                        y: 8,
                      }}
                      animate={{
                        opacity: 1,
                        y: 0,
                      }}
                    >
                      <div>
                        <span>
                          SELECTED OBJECT
                        </span>

                        <strong>
                          #
                          {String(
                            selectedObject.object_id,
                          ).padStart(2, "0")}
                        </strong>
                      </div>

                      <div>
                        <strong>
                          {selectedObject.accepted
                            ? humanizeLabel(
                                selectedObject
                                  .classifier
                                  ?.prediction,
                              )
                            : rejectionLabel(
                                selectedObject.rejection_reason,
                              )}
                        </strong>

                        <button
                          type="button"
                          onClick={clearSelection}
                          aria-label="Clear selection"
                        >
                          <X size={13} />
                        </button>
                      </div>
                    </motion.div>
                  )}
              </div>

              {activePrediction && (
                <div className="ww-vision-footer">
                  <div>
                    <span className="ww-status-dot ww-status-dot--accepted" />
                    Verified {acceptedDetections}
                  </div>

                  <div>
                    <span className="ww-status-dot ww-status-dot--rejected" />
                    Filtered {totalRejected}
                  </div>

                  <div className="ww-vision-footer-hint">
                    <MousePointer2 size={13} />
                    Hover or click a detection
                  </div>
                </div>
              )}

              {!activePrediction &&
                !isAnalyzing && (
                  <button
                    type="button"
                    className="ww-analyze-button"
                    onClick={analyzeWaste}
                  >
                    <Sparkles size={18} />
                    Analyze Waste
                    <span>↗</span>
                  </button>
                )}

              {isAnalyzing && (
                <motion.div
                  className="ww-analyzing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <Loader2
                    size={20}
                    className="ww-spin"
                  />

                  <div>
                    <strong>
                      Reading the image...
                    </strong>

                    <span>
                      Detecting objects → classifying →
                      verifying
                    </span>
                  </div>
                </motion.div>
              )}

              {error && (
                <motion.div
                  className="ww-error"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <AlertCircle size={19} />

                  <div>
                    <strong>Analysis failed</strong>
                    <span>{error}</span>
                  </div>
                </motion.div>
              )}
            </div>

            {/* ===================================================
                DECISION ENGINE
            =================================================== */}

            {activePrediction && (
              <div className="ww-results-panel">
                <div className="ww-results-heading">
                  <div>
                    <span>DECISION ENGINE</span>

                    <h3>Waste Intelligence</h3>
                  </div>

                  <div className="ww-analysis-status">
                    <span className="ww-live-dot" />
                    VERIFIED
                  </div>
                </div>

                {/* =================================================
                    ACCOUNTING
                ================================================= */}

                <div className="ww-accounting">
                  <motion.div
                    className="ww-accounting-primary"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <span>RAW PROPOSALS</span>
                    <strong>{rawDetections}</strong>
                    <small>YOLO11s candidates</small>
                  </motion.div>

                  <div className="ww-accounting-arrow">
                    →
                  </div>

                  <motion.div
                    className="ww-accounting-primary ww-accounting-primary--accepted"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{
                      opacity: 1,
                      y: 0,
                    }}
                    transition={{ delay: 0.08 }}
                  >
                    <span>VERIFIED</span>
                    <strong>{acceptedDetections}</strong>
                    <small>Trusted outputs</small>
                  </motion.div>

                  <motion.div
                    className="ww-accounting-primary ww-accounting-primary--rejected"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{
                      opacity: 1,
                      y: 0,
                    }}
                    transition={{ delay: 0.14 }}
                  >
                    <span>FILTERED</span>
                    <strong>{totalRejected}</strong>
                    <small>Safety rejections</small>
                  </motion.div>
                </div>

                {/* =================================================
                    FILTER BREAKDOWN
                ================================================= */}

                {totalRejected > 0 && (
                  <div className="ww-filter-breakdown">
                    <div className="ww-filter-heading">
                      <div>
                        <span>VERIFICATION LAYER</span>
                        <strong>
                          {totalRejected} filtered
                        </strong>
                      </div>

                      <ShieldCheck size={18} />
                    </div>

                    <div className="ww-filter-bars">
                      <div
                        className="ww-filter-bar ww-filter-bar--ood"
                        style={{
                          width:
                            rawDetections > 0
                              ? `${
                                  (oodRejected /
                                    rawDetections) *
                                  100
                                }%`
                              : "0%",
                        }}
                      />

                      <div
                        className="ww-filter-bar ww-filter-bar--compatibility"
                        style={{
                          width:
                            rawDetections > 0
                              ? `${
                                  (compatibilityRejected /
                                    rawDetections) *
                                  100
                                }%`
                              : "0%",
                        }}
                      />
                    </div>

                    <div className="ww-filter-legend">
                      <span>
                        <i className="ww-dot ww-dot--ood" />
                        OOD <b>{oodRejected}</b>
                      </span>

                      <span>
                        <i className="ww-dot ww-dot--compatibility" />
                        Compatibility{" "}
                        <b>{compatibilityRejected}</b>
                      </span>
                    </div>
                  </div>
                )}

                {/* =================================================
                    TRUSTED OUTPUTS
                ================================================= */}

                {acceptedObjects.length > 0 && (
                  <div className="ww-object-section">
                    <div className="ww-section-label">
                      <div>
                        <span>TRUSTED OUTPUTS</span>
                        <small>
                          Verified by classifier + safety gate
                        </small>
                      </div>

                      <b>{acceptedObjects.length}</b>
                    </div>

                    <div className="ww-object-list">
                      {acceptedObjects.map((item) => (
                        <ObjectCard
                          key={`accepted-${item.object_id}`}
                          item={item}
                          active={
                            hoveredObjectId ===
                              item.object_id ||
                            selectedObjectId ===
                              item.object_id
                          }
                          expanded={
                            expandedObjectId ===
                            item.object_id
                          }
                          onHover={setHoveredObjectId}
                          onSelect={selectObject}
                          onToggleDetails={(id) =>
                            setExpandedObjectId(
                              (current) =>
                                current === id
                                  ? null
                                  : id,
                            )
                          }
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* =================================================
                    NO RESULT
                ================================================= */}

                {acceptedObjects.length === 0 &&
                  rejectedObjects.length === 0 && (
                    <div className="ww-no-results">
                      <Trash2 size={26} />

                      <h4>No waste detected</h4>

                      <p>
                        WasteWise AI did not find any
                        candidate waste objects in this
                        image.
                      </p>
                    </div>
                  )}

                {/* =================================================
                    FILTERED OUTPUTS
                ================================================= */}

                {rejectedObjects.length > 0 && (
                  <div className="ww-object-section ww-object-section--rejected">
                    <div className="ww-section-label ww-section-label--rejected">
                      <div>
                        <span>FILTERED DETECTIONS</span>
                        <small>
                          Rejected before final decision
                        </small>
                      </div>

                      <b>{rejectedObjects.length}</b>
                    </div>

                    <div className="ww-object-list">
                      {rejectedObjects.map((item) => (
                        <ObjectCard
                          key={`rejected-${item.object_id}`}
                          item={item}
                          active={
                            hoveredObjectId ===
                              item.object_id ||
                            selectedObjectId ===
                              item.object_id
                          }
                          expanded={
                            expandedObjectId ===
                            item.object_id
                          }
                          onHover={setHoveredObjectId}
                          onSelect={selectObject}
                          onToggleDetails={(id) =>
                            setExpandedObjectId(
                              (current) =>
                                current === id
                                  ? null
                                  : id,
                            )
                          }
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* =================================================
                    PIPELINE
                ================================================= */}

                <div className="ww-model-card">
                  <div className="ww-model-header">
                    <div>
                      <span>PIPELINE TELEMETRY</span>
                      <strong>
                        WasteWise AI Stack
                      </strong>
                    </div>

                    <Sparkles size={18} />
                  </div>

                  <div className="ww-pipeline-flow">
                    <div>
                      <span>01</span>
                      <strong>YOLO11s</strong>
                      <small>Detection</small>
                    </div>

                    <div className="ww-pipeline-line">
                      →
                    </div>

                    <div>
                      <span>02</span>
                      <strong>ResNet50</strong>
                      <small>19 classes</small>
                    </div>

                    <div className="ww-pipeline-line">
                      →
                    </div>

                    <div>
                      <span>03</span>
                      <strong>OOD</strong>
                      <small>Verification</small>
                    </div>

                    <div className="ww-pipeline-line">
                      →
                    </div>

                    <div>
                      <span>04</span>
                      <strong>Decision</strong>
                      <small>Accept / reject</small>
                    </div>
                  </div>

                  <div className="ww-model-row">
                    <span>Classifier classes</span>
                    <strong>
                      {activePrediction.classifier
                        ?.class_count ?? 19}
                    </strong>
                  </div>

                  <div className="ww-model-row">
                    <span>OOD threshold</span>
                    <strong>
                      {activePrediction.classifier
                        ? activePrediction.classifier.ood_threshold.toFixed(
                            4,
                          )
                        : "0.2304"}
                    </strong>
                  </div>

                  <div className="ww-model-row">
                    <span>Detection threshold</span>
                    <strong>
                      {(
                        activePrediction.detector
                          ?.confidence_threshold ?? 0
                      ).toFixed(2)}
                    </strong>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* =====================================================
              ANALYSIS FOOTER
          ===================================================== */}

          {activePrediction && (
            <div className="ww-analysis-footer">
              <div className="ww-analysis-time">
                <Clock3 size={14} />

                <span>
                  Analysis complete
                  {activePrediction.duration_ms
                    ? ` in ${Math.round(
                        activePrediction.duration_ms,
                      )} ms`
                    : ""}
                </span>

                <span className="ww-analysis-valid">
                  <CheckCircle2 size={13} />
                  {activePrediction.accounting?.is_valid ??
                  activePrediction.summary.accounting_valid
                    ? "Accounting valid"
                    : "Check accounting"}
                </span>
              </div>

              <button
                type="button"
                className="ww-secondary-button"
                onClick={analyzeAnotherImage}
              >
                <ImageIcon size={15} />
                Analyze Another Image
                <span>↗</span>
              </button>
            </div>
          )}

          {!activePrediction &&
            selectedFile &&
            !isAnalyzing &&
            !error && (
              <div className="ww-pre-analysis-hint">
                <Zap size={15} />
                <span>
                  Image loaded. Start the perception pipeline
                  to detect and classify waste.
                </span>
              </div>
            )}
        </motion.div>
      )}
    </section>
  );
}