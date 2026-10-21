import {
  Camera,
  FileImage,
  ImagePlus,
  Upload,
  X,
} from "lucide-react";

import {
  useRef,
  useState,
} from "react";

interface UploadZoneProps {
  onFileSelected: (
    file: File,
  ) => void;

  onCameraClick: () => void;
}

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
];

const MAX_FILE_SIZE =
  10 * 1024 * 1024;

function UploadZone({
  onFileSelected,
  onCameraClick,
}: UploadZoneProps) {
  const inputRef =
    useRef<HTMLInputElement>(null);

  const [isDragging, setIsDragging] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const handleFile = (
    file?: File,
  ) => {
    if (!file) return;

    setError(null);

    if (
      !ACCEPTED_TYPES.includes(
        file.type,
      )
    ) {
      setError(
        "Please choose a JPG, PNG, or WebP image.",
      );
      return;
    }

    if (
      file.size >
      MAX_FILE_SIZE
    ) {
      setError(
        "Image size must be 10 MB or less.",
      );
      return;
    }

    onFileSelected(file);
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    handleFile(
      event.target.files?.[0],
    );

    event.target.value = "";
  };

  const handleDrop = (
    event: React.DragEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();

    setIsDragging(false);

    handleFile(
      event.dataTransfer.files?.[0],
    );
  };

  return (
    <div
      className={[
        "upload-zone",
        isDragging
          ? "is-dragging"
          : "",
      ].join(" ")}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() =>
        setIsDragging(false)
      }
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden-input"
        onChange={
          handleInputChange
        }
      />

      <div className="upload-grid">
        <div className="upload-copy">
          <div className="upload-icon">
            <ImagePlus
              size={27}
              strokeWidth={1.6}
            />
          </div>

          <div>
            <span className="upload-kicker">
              INPUT CHANNEL
            </span>

            <h3>
              Drop an image.
              <br />
              Let the model see.
            </h3>

            <p>
              Upload a clear waste image.
              Multiple objects are supported.
            </p>
          </div>
        </div>

        <div className="upload-actions">
          <button
            type="button"
            className="upload-primary"
            onClick={() =>
              inputRef.current?.click()
            }
            data-cursor="interactive"
          >
            <Upload size={17} />
            Choose image
          </button>

          <button
            type="button"
            className="upload-secondary"
            onClick={
              onCameraClick
            }
            data-cursor="interactive"
          >
            <Camera size={17} />
            Live camera
          </button>

          <div className="upload-formats">
            <FileImage size={14} />

            <span>
              JPG · PNG · WebP
            </span>

            <span>
              Max 10 MB
            </span>
          </div>
        </div>
      </div>

      <div className="upload-scan-line" />

      {isDragging && (
        <div className="drop-overlay">
          <div>
            <Upload size={25} />
            <strong>
              Release to analyze
            </strong>
          </div>
        </div>
      )}

      {error && (
        <div className="upload-error">
          <X size={15} />
          {error}
        </div>
      )}
    </div>
  );
}

export default UploadZone;