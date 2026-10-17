import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, explainImage, getHealth, predictImage } from "./api.js";

const LABELS = {
  little_or_none: "Little or none",
  mild: "Mild",
  severe: "Severe",
};

const CLASS_ORDER = ["little_or_none", "mild", "severe"];

const ACCEPT = "image/jpeg,image/png,image/webp,image/gif,image/bmp";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [cam, setCam] = useState(null);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [imageSize, setImageSize] = useState(null);
  const [lightbox, setLightbox] = useState(null);
  const [mobileView, setMobileView] = useState("both");
  const fileInputRef = useRef(null);
  const closeRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ ready: false, error: true }));
  }, []);

  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!lightbox) return undefined;
    const previous = document.activeElement;
    closeRef.current?.focus();
    function onKey(event) {
      if (event.key === "Escape") setLightbox(null);
    }
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      if (previous && typeof previous.focus === "function") previous.focus();
    };
  }, [lightbox]);

  function resetOutputs() {
    if (cam?.url) URL.revokeObjectURL(cam.url);
    setCam(null);
    setResult(null);
    setLightbox(null);
  }

  function applyFile(next) {
    resetOutputs();
    setImageSize(null);
    setFile(next || null);
    setError("");
  }

  function onFile(event) {
    applyFile(event.target.files?.[0] || null);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragOver(false);
    const dropped = event.dataTransfer.files?.[0];
    if (!dropped) return;
    if (dropped.type && !dropped.type.startsWith("image/")) {
      setError("Unsupported image. Use JPEG, PNG, WebP, GIF, or BMP.");
      return;
    }
    applyFile(dropped);
  }

  function resetAll() {
    resetOutputs();
    setFile(null);
    setImageSize(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function analyze() {
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setBusy("predict");
    setError("");
    resetOutputs();
    try {
      const prediction = await predictImage(file);
      setResult(prediction);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setBusy("");
    }
  }

  async function explain() {
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setBusy("explain");
    setError("");
    try {
      if (cam?.url) URL.revokeObjectURL(cam.url);
      const overlay = await explainImage(file);
      setCam(overlay);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setBusy("");
    }
  }

  const modelName = result?.model || health?.model;
  const deviceName = result?.device || health?.device;
  const lightboxSrc = lightbox === "source" ? previewUrl : lightbox === "cam" ? cam?.url : null;
  const lightboxLabel = lightbox === "source" ? "Source image" : "Activation overlay";

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <Mark />
          <div>
            <p className="brand-name">Fieldnote / Damage Lab</p>
            <p className="brand-sub">Visual Damage Assessment</p>
          </div>
        </div>
        <div className="mast-meta">
          <span className="badge">Research demo</span>
          <p className={`status ${health?.ready ? "ok" : "warn"}`} role="status">
            {health?.ready
              ? `Linked · ${health.model} · ${health.device}`
              : `Backend unavailable at ${API_BASE}`}
          </p>
        </div>
      </header>

      <section className="hero">
        <p className="sect">00 / Brief</p>
        <h1>Read the evidence. Assess the damage.</h1>
        <p className="lede">
          Examine disaster imagery through image-level severity classification and visual
          activation analysis.
        </p>
        <p className="disclaimer">
          Local research demonstration. This is not an operational emergency-response system.
          The model assigns one of three photograph-level labels: little or none, mild, or severe.
        </p>
      </section>

      <section className="intake" aria-labelledby="intake-heading">
        <p className="sect" id="intake-heading">
          01 / Evidence intake
        </p>
        <label
          className={`dropzone ${file ? "has-file" : ""} ${dragOver ? "over" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <input ref={fileInputRef} type="file" accept={ACCEPT} onChange={onFile} />
          <UploadMark />
          {file ? (
            <div className="drop-copy">
              <strong>Selected evidence</strong>
              <span className="filename">{file.name}</span>
              {imageSize ? (
                <span className="tech">
                  {imageSize.width} × {imageSize.height}
                </span>
              ) : null}
              <span>Click or drop another file to replace.</span>
            </div>
          ) : (
            <div className="drop-copy">
              <strong>Drop a disaster photograph here</strong>
              <span>Or browse files to select an image.</span>
              <span className="tech">JPEG · PNG · WebP · GIF · BMP</span>
            </div>
          )}
        </label>

        <div className="actions">
          <button type="button" onClick={analyze} disabled={Boolean(busy) || !file}>
            {busy === "predict" ? "Analyzing…" : "Analyze damage"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={explain}
            disabled={Boolean(busy) || !file}
          >
            {busy === "explain" ? "Generating overlay…" : "Generate Grad-CAM"}
          </button>
          <button type="button" className="ghost" onClick={resetAll} disabled={Boolean(busy)}>
            Reset
          </button>
        </div>

        <p className="sr-only" aria-live="polite">
          {busy === "predict"
            ? "Analyzing image"
            : busy === "explain"
              ? "Generating Grad-CAM"
              : error
                ? error
                : result
                  ? `Predicted ${LABELS[result.predicted_class] || result.predicted_class}`
                  : ""}
        </p>
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        {busy ? (
          <p className="busy-note pulse">
            {busy === "predict" ? "Running image-level classification…" : "Building Grad-CAM overlay…"}
          </p>
        ) : null}
      </section>

      <section className="evidence" aria-labelledby="evidence-heading">
        <div className="section-row">
          <p className="sect" id="evidence-heading">
            02 / Visual evidence
          </p>
          <div className="view-toggle" role="group" aria-label="Evidence view">
            <button
              type="button"
              className={mobileView === "source" ? "on" : ""}
              onClick={() => setMobileView("source")}
            >
              Source
            </button>
            <button
              type="button"
              className={mobileView === "overlay" ? "on" : ""}
              onClick={() => setMobileView("overlay")}
            >
              Overlay
            </button>
            <button
              type="button"
              className={mobileView === "both" ? "on" : ""}
              onClick={() => setMobileView("both")}
            >
              Both
            </button>
          </div>
        </div>

        <div className={`frames view-${mobileView}`}>
          <figure className="frame source-frame">
            <figcaption>
              <span>Source image</span>
              {file ? <span className="caption-file">{file.name}</span> : null}
            </figcaption>
            <div className="canvas">
              {previewUrl ? (
                <button
                  type="button"
                  className="enlarge"
                  onClick={() => setLightbox("source")}
                  aria-label="Enlarge source image"
                >
                  <img
                    src={previewUrl}
                    alt="Selected disaster photograph"
                    onLoad={(event) => {
                      setImageSize({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight,
                      });
                    }}
                  />
                </button>
              ) : (
                <p className="empty-copy">No photograph submitted.</p>
              )}
            </div>
          </figure>

          <figure className="frame overlay-frame">
            <figcaption>
              <span>Activation overlay</span>
              {cam?.predictedClass ? (
                <span className="caption-file">{cam.predictedClass}</span>
              ) : null}
            </figcaption>
            <div className="canvas overlay-canvas">
              {cam?.url ? (
                <button
                  type="button"
                  className="enlarge"
                  onClick={() => setLightbox("cam")}
                  aria-label="Enlarge Grad-CAM overlay"
                >
                  <img src={cam.url} alt="Grad-CAM overlay for the predicted class" />
                </button>
              ) : (
                <p className="empty-copy">
                  {file
                    ? "Generate Grad-CAM to inspect activations for the predicted class."
                    : "Overlay appears after an explanation is requested."}
                </p>
              )}
            </div>
          </figure>
        </div>
        <p className="note">
          {cam?.disclaimer ||
            "Grad-CAM highlights image regions associated with the predicted class. It does not prove causal reasoning."}
        </p>
      </section>

      <section className="report" aria-labelledby="report-heading">
        <p className="sect" id="report-heading">
          03 / Assessment report
        </p>

        {!result && busy !== "predict" ? (
          <p className="empty-copy report-empty">
            {file
              ? "Analyze the photograph to produce an image-level severity assessment."
              : "Submit evidence, then analyze to record severity, confidence, and class probabilities."}
          </p>
        ) : null}

        {busy === "predict" ? <p className="empty-copy pulse">Writing assessment…</p> : null}

        {result ? (
          <div className="report-body">
            <div className={`finding ${result.predicted_class}`}>
              <p className="finding-kicker">Predicted severity</p>
              <p className="finding-value">
                {LABELS[result.predicted_class] || result.predicted_class}
              </p>
              <p className="finding-mark">
                Class {CLASS_ORDER.indexOf(result.predicted_class) + 1} of 3 · photograph-level label
              </p>
            </div>

            <dl className="metrics">
              <div>
                <dt>Confidence</dt>
                <dd>{(result.confidence * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Inference</dt>
                <dd>{result.inference_time_ms.toFixed(1)} ms</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{modelName || "—"}</dd>
              </div>
              {deviceName ? (
                <div>
                  <dt>Device</dt>
                  <dd>{deviceName}</dd>
                </div>
              ) : null}
            </dl>

            <div className="dist">
              <h3>Class probabilities</h3>
              <ul className="bars">
                {CLASS_ORDER.map((name) => {
                  const value = result.probabilities?.[name];
                  if (value == null) return null;
                  const pct = (value * 100).toFixed(1);
                  const active = name === result.predicted_class;
                  return (
                    <li key={name} className={`${name} ${active ? "active" : ""}`}>
                      <span>{LABELS[name]}</span>
                      <div className="bar" aria-hidden="true">
                        <div className="fill" style={{ width: `${pct}%` }} />
                      </div>
                      <em>{pct}%</em>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        ) : null}
      </section>

      <footer className="colophon">
        <p>Fieldnote / Damage Lab</p>
        <p>
          Research demonstration for image-level disaster damage severity classification.
          {modelName ? ` Local inference · ${modelName}.` : null}
        </p>
      </footer>

      {lightboxSrc ? (
        <div
          className="lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={lightboxLabel}
          onClick={() => setLightbox(null)}
        >
          <div className="lightbox-inner" onClick={(event) => event.stopPropagation()}>
            <p className="sect">{lightboxLabel}</p>
            <img src={lightboxSrc} alt={lightboxLabel} />
            <button
              ref={closeRef}
              type="button"
              className="secondary"
              onClick={() => setLightbox(null)}
            >
              Close
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="3" y="3" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.25" />
      <circle cx="16" cy="16" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.25" />
      <path d="M16 3v5.5M16 23.5V29M3 16h5.5M23.5 16H29" stroke="currentColor" strokeWidth="1.25" />
    </svg>
  );
}

function UploadMark() {
  return (
    <svg className="upload-mark" viewBox="0 0 40 40" aria-hidden="true">
      <rect x="6" y="8" width="28" height="24" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <path d="M14 24l6-7 4 5 3-3 5 5" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="16" cy="15" r="1.8" fill="currentColor" />
    </svg>
  );
}

function formatError(err) {
  const message = err?.message || String(err);
  if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
    return "Cannot reach the API. Is FastAPI running?";
  }
  return message;
}
