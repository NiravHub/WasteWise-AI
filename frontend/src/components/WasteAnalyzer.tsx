import { useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Loader2,
  Recycle,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  XCircle,
  Zap,
} from "lucide-react";

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

interface ObjectCardProps {
  item: WasteObject;
  active: boolean;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
}

function formatPercent(value: number) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return "0.0%";
  }

  return `${Math.max(
    0,
    Math.min(100, numericValue),
  ).toFixed(1)}%`;
}

function humanizeLabel(
  value: string | null | undefined,
) {
  if (!value) return "Unknown";

  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) =>
      char.toUpperCase(),
    );
}

function confidenceClass(confidence: number) {
  if (confidence >= 85) return "high";
  if (confidence >= 60) return "medium";
  return "low";
}

function getWasteIcon(
  label: string | null | undefined,
) {
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

function rejectionLabel(
  reason: string | null | undefined,
) {
  switch (reason) {
    case "ood_rejection":
      return "OOD rejection";

    case "compatibility_mismatch":
      return "Compatibility mismatch";

    default:
      return "Pipeline rejection";
  }
}

function rejectionDescription(
  reason: string | null | undefined,
) {
  switch (reason) {
    case "ood_rejection":
      return "The classifier considered this detection outside the supported visual distribution.";

    case "compatibility_mismatch":
      return "The detector and classifier predictions were semantically incompatible.";

    default:
      return "This detection did not pass the final WasteWise verification layer.";
  }
}

function ObjectCard({
  item,
  active,
  onHover,
  onSelect,
}: ObjectCardProps) {
  const accepted = item.accepted !== false;
  const prediction =
    item.classifier?.prediction ?? null;

  const confidence =
    item.classifier?.confidence ?? 0;

  const Icon = getWasteIcon(prediction);

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLElement>,
  ) => {
    if (
      event.key === "Enter" ||
      event.key === " "
    ) {
      event.preventDefault();
      onSelect(item.object_id);
    }
  };

  if (!accepted) {
    return (
      <article
        className={`ww-object-card ww-object-card--rejected ${
          active ? "is-active" : ""
        }`}
        onMouseEnter={() =>
          onHover(item.object_id)
        }
        onMouseLeave={() => onHover(null)}
        onClick={() =>
          onSelect(item.object_id)
        }
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        data-object-id={item.object_id}
      >
        <div className="ww-object-card-header">
          <div className="ww-object-title">
            <div className="ww-object-icon ww-object-icon--rejected">
              <XCircle size={18} />
            </div>

            <div>
              <span className="ww-object-number">
                Object {item.object_id}
              </span>

              <h4>Filtered detection</h4>
            </div>
          </div>

          <div className="ww-rejection-badge">
            FILTERED
          </div>
        </div>

        <div className="ww-rejection-reason">
          <div className="ww-rejection-reason-title">
            <AlertCircle size={16} />

            <strong>
              {rejectionLabel(
                item.rejection_reason,
              )}
            </strong>
          </div>

          <p>
            {rejectionDescription(
              item.rejection_reason,
            )}
          </p>
        </div>

        <div className="ww-object-grid">
          <div>
            <span>Detector</span>

            <strong>
              {humanizeLabel(
                item.detector?.class,
              )}
            </strong>

            <small>
              {formatPercent(
                item.detector?.confidence ?? 0,
              )}
            </small>
          </div>

          <div>
            <span>Classifier</span>

            <strong>
              {prediction
                ? humanizeLabel(prediction)
                : "Unsupported"}
            </strong>

            <small>
              {formatPercent(confidence)}
            </small>
          </div>

          <div>
            <span>OOD</span>

            <strong>
              {item.classifier?.is_supported
                ? "Supported"
                : "Rejected"}
            </strong>

            <small>
              Similarity{" "}
              {Number(
                item.classifier
                  ?.ood_similarity ?? 0,
              ).toFixed(3)}
            </small>
          </div>

          <div>
            <span>Decision</span>

            <strong className="ww-unsupported">
              Rejected
            </strong>

            <small>
              {rejectionLabel(
                item.rejection_reason,
              )}
            </small>
          </div>
        </div>
      </article>
    );
  }

  return (
    <article
      className={`ww-object-card ww-object-card--accepted ${
        active ? "is-active" : ""
      }`}
      onMouseEnter={() =>
        onHover(item.object_id)
      }
      onMouseLeave={() => onHover(null)}
      onClick={() =>
        onSelect(item.object_id)
      }
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      data-object-id={item.object_id}
    >
      <div className="ww-object-card-header">
        <div className="ww-object-title">
          <div className="ww-object-icon">
            <Icon size={18} />
          </div>

          <div>
            <span className="ww-object-number">
              Object {item.object_id}
            </span>

            <h4>
              {humanizeLabel(prediction)}
            </h4>
          </div>
        </div>

        <div
  className={`ww-confidence ${confidenceClass(
    confidence,
  )}`}
  title={`Classification confidence: ${formatPercent(confidence)}`}
>
  <span>{formatPercent(confidence)}</span>

  <i
    style={{
      width: `${Math.max(
        0,
        Math.min(100, confidence * 100),
      )}%`,
    }}
  />
</div>
        </div>

      <div className="ww-object-status">
        <CheckCircle2 size={14} />
        VERIFIED BY WASTEWISE
      </div>

      <div className="ww-object-grid">
        <div>
          <span>Detector</span>

          <strong>
            {humanizeLabel(
              item.detector?.class,
            )}
          </strong>

          <small>
            {formatPercent(
              item.detector?.confidence ?? 0,
            )}
          </small>
        </div>

        <div>
          <span>Classifier</span>

          <strong>
            {humanizeLabel(prediction)}
          </strong>

          <small>
            {humanizeLabel(
              item.classifier
                ?.confidence_level,
            )}
          </small>
        </div>

        <div>
          <span>Category</span>

          <strong>
            {item.classifier?.category ??
              "Unknown"}
          </strong>
        </div>

        <div>
          <span>OOD Check</span>

          <strong className="ww-supported">
            Supported
          </strong>

          <small>
            Similarity{" "}
            {Number(
              item.classifier
                ?.ood_similarity ?? 0,
            ).toFixed(3)}
          </small>
        </div>
      </div>

      {item.classifier?.recommendation && (
        <div className="ww-recommendation">
          <Recycle size={16} />

          <span>
            {item.classifier.recommendation}
          </span>
        </div>
      )}

      {item.classifier?.top_predictions
        ?.length > 0 && (
        <div className="ww-top-predictions">
          <span>
            Alternative predictions
          </span>

          <div>
            {item.classifier.top_predictions
              .slice(0, 3)
              .map((predictionItem) => (
                <span
                  key={`${item.object_id}-${predictionItem.class}`}
                >
                  {humanizeLabel(
                    predictionItem.class,
                  )}{" "}
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
    </article>
  );
}

export default function WasteAnalyzer({
  selectedFile,
  previewUrl,
  onClear,
}: WasteAnalyzerProps) {
  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(
      null,
    );

  const [isAnalyzing, setIsAnalyzing] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const [hoveredObjectId, setHoveredObjectId] =
    useState<number | null>(null);

  const [selectedObjectId, setSelectedObjectId] =
    useState<number | null>(null);

  const requestIdRef = useRef(0);

  const activePrediction =
    prediction &&
    selectedFile &&
    prediction.filename ===
      selectedFile.name
      ? prediction
      : null;

  const analyzeWaste = async () => {
    if (!selectedFile) return;

    const fileBeingAnalyzed =
      selectedFile;

    const requestId =
      ++requestIdRef.current;

    setIsAnalyzing(true);
    setError(null);
    setPrediction(null);
    setHoveredObjectId(null);
    setSelectedObjectId(null);

    try {
      const result = await predictWaste(
        fileBeingAnalyzed,
      );

      if (
        requestId !==
        requestIdRef.current
      ) {
        return;
      }

      setPrediction(result);
    } catch (err) {
      if (
        requestId !==
        requestIdRef.current
      ) {
        return;
      }

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the image.",
      );
    } finally {
      if (
        requestId ===
        requestIdRef.current
      ) {
        setIsAnalyzing(false);
      }
    }
  };

  const acceptedObjects =
    activePrediction?.objects ?? [];

  const rejectedObjects =
    activePrediction?.rejected_objects ??
    [];

  const rawDetections =
    activePrediction?.summary
      ?.raw_detections ??
    activePrediction?.detector
      ?.raw_detections ??
    activePrediction?.detector
      ?.detections ??
    acceptedObjects.length +
      rejectedObjects.length;

  const acceptedDetections =
    activePrediction?.summary
      ?.accepted_objects ??
    activePrediction?.detector
      ?.accepted_detections ??
    acceptedObjects.length;

  const totalRejected =
    activePrediction?.summary
      ?.total_rejections ??
    rejectedObjects.length;

  const oodRejected =
    activePrediction?.summary
      ?.ood_rejections ??
    rejectedObjects.filter(
      (item) =>
        item.rejection_reason ===
        "ood_rejection",
    ).length;

  const compatibilityRejected =
    activePrediction?.summary
      ?.compatibility_rejections ??
    rejectedObjects.filter(
      (item) =>
        item.rejection_reason ===
        "compatibility_mismatch",
    ).length;

  const overlayObjects =
    useMemo<OverlayObject[]>(() => {
      if (!activePrediction) return [];

      const accepted =
        activePrediction.objects ?? [];

      const rejected =
        activePrediction.rejected_objects ??
        [];

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

      const imageWidth =
        Number(
          activePrediction.image?.width,
        ) || 1;

      const imageHeight =
        Number(
          activePrediction.image?.height,
        ) || 1;

      return combined.map((item) => {
        const box = item.bounding_box;

        const left =
          (box.x1 / imageWidth) * 100;

        const top =
          (box.y1 / imageHeight) * 100;

        const width =
          ((box.x2 - box.x1) /
            imageWidth) *
          100;

        const height =
          ((box.y2 - box.y1) /
            imageHeight) *
          100;

        return {
          ...item,
          left: Math.max(
            0,
            Math.min(100, left),
          ),
          top: Math.max(
            0,
            Math.min(100, top),
          ),
          width: Math.max(
            0,
            Math.min(
              100 - left,
              width,
            ),
          ),
          height: Math.max(
            0,
            Math.min(
              100 - top,
              height,
            ),
          ),
        };
      });
    }, [activePrediction]);

  const clearAnalysis = () => {
    requestIdRef.current += 1;

    setPrediction(null);
    setError(null);
    setIsAnalyzing(false);
    setHoveredObjectId(null);
    setSelectedObjectId(null);

    onClear();
  };

  const handleSelectObject = (
    objectId: number,
  ) => {
    setSelectedObjectId((current) =>
      current === objectId
        ? null
        : objectId,
    );
  };

  const activeObjectId =
    selectedObjectId ??
    hoveredObjectId;

  return (
    <section className="ww-analyzer">
      {!selectedFile && (
        <div className="ww-empty-state">
          <div className="ww-empty-icon">
            <ScanSearch size={30} />
          </div>

          <div className="ww-empty-copy">
            <span className="ww-empty-eyebrow">
              READY STATE
            </span>

            <h3>
              Your waste intelligence
              console is ready.
            </h3>

            <p>
              Upload an image and
              WasteWise will detect,
              classify, verify, and
              explain each candidate
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
        </div>
      )}

      {selectedFile && (
        <div className="ww-analysis-workspace">
          {/* =====================================================
              IMAGE / VISION PANEL
          ===================================================== */}

          <div className="ww-preview-panel">
            <div className="ww-panel-heading">
              <div>
                <span>VISION INPUT</span>

                <h3>
                  {selectedFile.name}
                </h3>
              </div>

              <button
                type="button"
                className="ww-icon-button"
                onClick={clearAnalysis}
                title="Remove image"
                aria-label="Remove image"
              >
                <XCircle size={20} />
              </button>
            </div>

            <div className="ww-image-meta">
              <span>
                <span className="ww-live-dot" />
                IMAGE STREAM
              </span>

              {activePrediction && (
                <span>
                  {
                    activePrediction.image
                      .width
                  }{" "}
                  ×{" "}
                  {
                    activePrediction.image
                      .height
                  }
                </span>
              )}
            </div>

            <div className="ww-image-stage">
              {previewUrl ? (
                <div className="ww-image-wrapper">
                  <img
                    src={previewUrl}
                    alt="Waste selected for analysis"
                    draggable={false}
                  />

                  {activePrediction &&
                    overlayObjects.map(
                      (item) => {
                        const isActive =
                          activeObjectId ===
                          item.object_id;

                        const hasActiveObject =
                          activeObjectId !==
                          null;

                        const isDimmed =
                          hasActiveObject &&
                          !isActive;

                        const label =
                          item.accepted
                            ? `${humanizeLabel(
                                item.classifier
                                  ?.prediction,
                              )} ${formatPercent(
                                item
                                  .classifier
                                  ?.confidence ??
                                  0,
                              )}`
                            : `FILTERED · OBJECT ${item.object_id}`;

                        return (
                          <div
                            key={`overlay-${item.object_id}`}
                            className={[
  "ww-detection-box",
  item.accepted
    ? "ww-detection-box--accepted"
    : "ww-detection-box--rejected",
  item.top < 8
    ? "ww-detection-box--top-edge"
    : "",
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
                            role="button"
                            tabIndex={0}
                            aria-label={`Object ${item.object_id}: ${label}`}
                            onMouseEnter={() =>
                              setHoveredObjectId(
                                item.object_id,
                              )
                            }
                            onMouseLeave={() =>
                              setHoveredObjectId(
                                null,
                              )
                            }
                            onClick={() =>
                              handleSelectObject(
                                item.object_id,
                              )
                            }
                            onKeyDown={(
                              event,
                            ) => {
                              if (
                                event.key ===
                                  "Enter" ||
                                event.key ===
                                  " "
                              ) {
                                event.preventDefault();

                                handleSelectObject(
                                  item.object_id,
                                );
                              }
                            }}
                          >
                            <span>
                              <b>
                                {String(
                                  item.object_id,
                                ).padStart(
                                  2,
                                  "0",
                                )}
                              </b>

                              {label}
                            </span>
                          </div>
                        );
                      },
                    )}

                  {activePrediction &&
                    overlayObjects.length >
                      0 && (
                      <div className="ww-image-scanline" />
                    )}
                </div>
              ) : (
                <div className="ww-no-preview">
                  <Upload size={28} />

                  <span>
                    Preview unavailable
                  </span>
                </div>
              )}
            </div>

            {activePrediction && (
              <div className="ww-vision-footer">
                <span>
                  <Zap size={14} />
                  COMPUTER VISION ACTIVE
                </span>

                <span>
                  {rawDetections} proposals
                </span>
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
              <div className="ww-analyzing">
                <Loader2
                  size={20}
                  className="ww-spin"
                />

                <div>
                  <strong>
                    Reading the image...
                  </strong>

                  <span>
                    Detecting objects →
                    classifying →
                    verifying
                  </span>
                </div>
              </div>
            )}

            {error && (
              <div className="ww-error">
                <AlertCircle size={19} />

                <div>
                  <strong>
                    Analysis failed
                  </strong>

                  <span>
                    {error}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* =====================================================
              INTELLIGENCE PANEL
          ===================================================== */}

          {activePrediction && (
  <div
    className="ww-results-panel"
  >
              <div className="ww-results-heading">
                <div>
                  <span>
                    DECISION ENGINE
                  </span>

                  <h3>
                    Waste Intelligence
                  </h3>
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
                <div className="ww-accounting-primary">
                  <span>
                    RAW PROPOSALS
                  </span>

                  <strong>
                    {rawDetections}
                  </strong>
                </div>

                <div className="ww-accounting-arrow">
                  →
                </div>

                <div className="ww-accounting-primary ww-accounting-primary--accepted">
                  <span>
                    VERIFIED
                  </span>

                  <strong>
                    {acceptedDetections}
                  </strong>
                </div>

                <div className="ww-accounting-primary ww-accounting-primary--rejected">
                  <span>
                    FILTERED
                  </span>

                  <strong>
                    {totalRejected}
                  </strong>
                </div>
              </div>

              {/* =================================================
                  SAFETY FILTER
              ================================================= */}

              {totalRejected > 0 && (
                <div className="ww-filter-breakdown">
                  <div className="ww-filter-heading">
                    <span>
                      SAFETY FILTER
                    </span>

                    <span>
                      {totalRejected}{" "}
                      rejected
                    </span>
                  </div>

                  <div className="ww-filter-bars">
                    <div
                      className="ww-filter-bar"
                      style={{
                        width:
                          rawDetections >
                          0
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
                          rawDetections >
                          0
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
                      OOD{" "}
                      {oodRejected}
                    </span>

                    <span>
                      <i className="ww-dot ww-dot--compatibility" />
                      Compatibility{" "}
                      {
                        compatibilityRejected
                      }
                    </span>
                  </div>
                </div>
              )}

              <div
                className="ww-results-content"
                data-lenis-prevent
                onWheel={(event) => {
                  const content = event.currentTarget;

                  if (content.scrollHeight <= content.clientHeight) {
                    return;
                  }

                  event.preventDefault();
                  event.stopPropagation();

                  const nextScrollTop =
                    content.scrollTop + event.deltaY;

                  content.scrollTop = Math.max(
                    0,
                    Math.min(
                      nextScrollTop,
                      content.scrollHeight - content.clientHeight,
                    ),
                  );
                }}
              >
              {/* =================================================
                  OBJECTS
              ================================================= */}

              {acceptedObjects.length ===
                0 &&
              rejectedObjects.length ===
                0 ? (
                <div className="ww-no-results">
                  <Trash2 size={26} />

                  <h4>
                    No waste detected
                  </h4>

                  <p>
                    WasteWise AI did not
                    find any candidate
                    waste objects in this
                    image.
                  </p>
                </div>
              ) : (
                <>
                  {acceptedObjects.length >
                    0 && (
                    <div className="ww-object-section">
                      <div className="ww-section-label">
                        <span>
                          VERIFIED OBJECTS
                        </span>

                        <b>
                          {
                            acceptedObjects.length
                          }
                        </b>
                      </div>

                      <div className="ww-object-list">
                        {acceptedObjects.map(
                          (item) => (
                            <ObjectCard
                              key={`accepted-${item.object_id}`}
                              item={item}
                              active={
                                activeObjectId ===
                                item.object_id
                              }
                              onHover={
                                setHoveredObjectId
                              }
                              onSelect={
                                handleSelectObject
                              }
                            />
                          ),
                        )}
                      </div>
                    </div>
                  )}

                  {rejectedObjects.length >
                    0 && (
                    <div className="ww-object-section ww-object-section--rejected">
                      <div className="ww-section-label ww-section-label--rejected">
                        <span>
                          FILTERED DETECTIONS
                        </span>

                        <b>
                          {
                            rejectedObjects.length
                          }
                        </b>
                      </div>

                      <div className="ww-object-list">
                        {rejectedObjects.map(
                          (item) => (
                            <ObjectCard
                              key={`rejected-${item.object_id}`}
                              item={item}
                              active={
                                activeObjectId ===
                                item.object_id
                              }
                              onHover={
                                setHoveredObjectId
                              }
                              onSelect={
                                handleSelectObject
                              }
                            />
                          ),
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* =================================================
                  PIPELINE TELEMETRY
              ================================================= */}

              <div className="ww-model-card">
                <div className="ww-model-header">
                  <div>
                    <span>
                      PIPELINE TELEMETRY
                    </span>

                    <strong>
                      WasteWise AI Stack
                    </strong>
                  </div>

                  <Sparkles size={18} />
                </div>

                <div className="ww-pipeline-flow">
                  <div>
                    <span>01</span>

                    <strong>
                      YOLO11s
                    </strong>

                    <small>
                      Detection
                    </small>
                  </div>

                  <div className="ww-pipeline-line">
                    →
                  </div>

                  <div>
                    <span>02</span>

                    <strong>
                      ResNet50
                    </strong>

                    <small>
                      19 classes
                    </small>
                  </div>

                  <div className="ww-pipeline-line">
                    →
                  </div>

                  <div>
                    <span>03</span>

                    <strong>
                      OOD
                    </strong>

                    <small>
                      Verify
                    </small>
                  </div>

                  <div className="ww-pipeline-line">
                    →
                  </div>

                  <div>
                    <span>04</span>

                    <strong>
                      Decision
                    </strong>

                    <small>
                      Accept / reject
                    </small>
                  </div>
                </div>

                <div className="ww-model-row">
                  <span>
                    Classifier classes
                  </span>

                  <strong>
                    {activePrediction
                      .classifier
                      ?.class_count ?? 19}
                  </strong>
                </div>

                <div className="ww-model-row">
                  <span>
                    OOD threshold
                  </span>

                  <strong>
                    {activePrediction
                      .classifier
                      ? activePrediction.classifier.ood_threshold.toFixed(
                          4,
                        )
                      : "0.2304"}
                  </strong>
                </div>

                <div className="ww-model-row">
                  <span>
                    Detection threshold
                  </span>

                  <strong>
                    {(
                      activePrediction
                        .detector
                        ?.confidence_threshold ??
                      0
                    ).toFixed(2)}
                  </strong>
                </div>
              </div>

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
                  Accounting valid
                </span>
              </div>

              <button
                type="button"
                className="ww-secondary-button"
                onClick={clearAnalysis}
              >
                Analyze Another Image
                <span>↗</span>
              </button>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}