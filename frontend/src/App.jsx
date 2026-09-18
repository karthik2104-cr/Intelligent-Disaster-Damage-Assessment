import { useEffect, useMemo, useState } from "react";
import { API_BASE, explainImage, getHealth, predictImage } from "./api.js";

const LABELS = {
  little_or_none: "Little or none",
  mild: "Mild",
  severe: "Severe",
};

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [cam, setCam] = useState(null);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

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

  function onFile(event) {
    const next = event.target.files?.[0];
    resetOutputs();
    setFile(next || null);
    setError("");
  }

  function resetOutputs() {
    if (cam?.url) URL.revokeObjectURL(cam.url);
    setCam(null);
    setResult(null);
  }

  function resetAll() {
    resetOutputs();
    setFile(null);
    setError("");
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

  return (
    <div className="page">
      <header>
        <p className="kicker">AI-assisted image classification</p>
        <h1>Disaster damage assessment</h1>
        <p className="lede">
          Upload a ground-level disaster photograph. EfficientNet-B0 estimates
          damage severity as <strong>little or none</strong>, <strong>mild</strong>, or{" "}
          <strong>severe</strong>. This is a research demo, not an operational
          response tool.
        </p>
      </header>

      <section className="status">
        {health?.ready ? (
          <span>
            API connected · {health.model} · {health.device} · {API_BASE}
          </span>
        ) : (
          <span className="warn">
            Backend unavailable at {API_BASE}. Start FastAPI, then refresh.
          </span>
        )}
      </section>

      <section className="card">
        <label className="file">
          <input type="file" accept="image/jpeg,image/png,image/webp,image/gif,image/bmp" onChange={onFile} />
          <span>{file ? file.name : "Select an image"}</span>
        </label>
        <div className="actions">
          <button type="button" onClick={analyze} disabled={Boolean(busy) || !file}>
            {busy === "predict" ? "Analyzing…" : "Analyze damage"}
          </button>
          <button type="button" className="secondary" onClick={explain} disabled={Boolean(busy) || !file}>
            {busy === "explain" ? "Explaining…" : "Show Grad-CAM"}
          </button>
          <button type="button" className="ghost" onClick={resetAll} disabled={Boolean(busy)}>
            Reset
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      <section className="grid">
        <figure>
          <figcaption>Input</figcaption>
          {previewUrl ? <img src={previewUrl} alt="Upload preview" /> : <div className="placeholder">No image</div>}
        </figure>
        <figure>
          <figcaption>Grad-CAM overlay</figcaption>
          {cam?.url ? <img src={cam.url} alt="Grad-CAM overlay" /> : <div className="placeholder">Not requested</div>}
        </figure>
      </section>
      {cam?.disclaimer ? <p className="note">{cam.disclaimer}</p> : <p className="note">Grad-CAM highlights activations for the predicted class. It is not causal proof.</p>}

      {result ? (
        <section className="card result">
          <h2>
            {LABELS[result.predicted_class] || result.predicted_class}{" "}
            <span className={`pill ${result.predicted_class}`}>{result.predicted_class}</span>
          </h2>
          <p>
            Confidence {(result.confidence * 100).toFixed(1)}% · {result.inference_time_ms.toFixed(1)} ms ·{" "}
            {result.model}
          </p>
          <ul className="bars">
            {Object.entries(result.probabilities).map(([name, value]) => (
              <li key={name}>
                <span>{LABELS[name] || name}</span>
                <div className="bar">
                  <div style={{ width: `${(value * 100).toFixed(1)}%` }} />
                </div>
                <em>{(value * 100).toFixed(1)}%</em>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function formatError(err) {
  const message = err?.message || String(err);
  if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
    return "Cannot reach the API. Is FastAPI running?";
  }
  return message;
}
