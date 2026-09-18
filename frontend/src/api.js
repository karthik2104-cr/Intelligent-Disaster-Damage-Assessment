export const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function parseError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function predictImage(file) {
  const form = new FormData();
  form.append("image", file, file.name);
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function explainImage(file) {
  const form = new FormData();
  form.append("image", file, file.name);
  const response = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  const blob = await response.blob();
  return {
    url: URL.createObjectURL(blob),
    predictedClass: response.headers.get("X-Predicted-Class"),
    disclaimer: response.headers.get("X-GradCAM-Disclaimer"),
  };
}
