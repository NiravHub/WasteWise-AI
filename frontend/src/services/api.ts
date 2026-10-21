export interface TopPrediction {
  class: string;
  confidence: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface DetectorObject {
  class: string;
  confidence: number;
}

export interface ClassifierObject {
  is_supported: boolean;
  prediction: string | null;
  confidence: number;
  confidence_level: string;
  category: string;
  recommendation: string;
  ood_similarity: number;
  ood_threshold: number;
  ood_best_class: string;
  top_predictions: TopPrediction[];
}

export interface WasteObject {
  object_id: number;
  bounding_box: BoundingBox;
  detector: DetectorObject;
  classifier: ClassifierObject;

  // Present for accepted/rejected object accounting
  accepted?: boolean;
  rejection_reason?: string | null;
}

export interface DetectorInfo {
  model: string;
  confidence_threshold: number;
  classes: string[];

  // Production accounting
  raw_detections: number;
  accepted_detections: number;

  // Legacy/backward-compatible field
  detections?: number;
}

export interface ClassifierInfo {
  model: string;
  classes: string[];
  class_count: number;
  ood_calibration: string;
  ood_threshold: number;
}

export interface ImageInfo {
  width: number;
  height: number;
}

export interface Summary {
  raw_detections: number;
  total_objects: number;
  accepted_objects: number;
  supported_objects: number;
  unsupported_objects: number;

  ood_rejections: number;
  compatibility_rejections: number;
  other_rejections: number;
  total_rejections: number;

  accounting_valid: boolean;
}

export interface Accounting {
  raw_detections: number;
  accepted_detections: number;

  ood_rejected: number;
  compatibility_rejected: number;
  other_rejected: number;

  total_rejected: number;
  is_valid: boolean;
}

export interface PredictionResponse {
  success: boolean;
  filename: string;

  image: ImageInfo;

  detector: DetectorInfo;

  classifier: ClassifierInfo | null;

  // Final accepted objects
  objects: WasteObject[];

  // Objects rejected by OOD / semantic compatibility / other rules
  rejected_objects?: WasteObject[];

  summary: Summary;

  accounting?: Accounting;
  duration_ms?: number;
}

const API_BASE_URL = "http://127.0.0.1:8000";

export async function predictWaste(
  file: File
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Prediction failed (${response.status})`;

    try {
      const errorData = await response.json();

      if (errorData?.detail) {
        message =
          typeof errorData.detail === "string"
            ? errorData.detail
            : JSON.stringify(errorData.detail);
      }
    } catch {
      // Keep the default HTTP error message
    }

    throw new Error(message);
  }

  const data = (await response.json()) as PredictionResponse;

  if (!data.success) {
    throw new Error("WasteWise prediction failed.");
  }

  return data;
}