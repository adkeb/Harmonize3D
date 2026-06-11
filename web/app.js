const state = {
  config: null,
  activeJobId: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

function setImage(id, url) {
  const image = $(id);
  if (!url) {
    image.removeAttribute("src");
    return;
  }
  image.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

function setStatus(job) {
  const status = job?.status || "idle";
  $("statusPill").textContent = status;
  $("statusPill").className = `status-pill ${status}`;
  $("jobTitle").textContent = job?.id || "待命";
  $("progressBar").style.width = `${Math.round((job?.progress || 0) * 100)}%`;
  document.querySelectorAll("[data-stage]").forEach((node) => {
    node.classList.toggle("active", node.dataset.stage === job?.stage);
  });
  $("runButton").disabled = status === "running";
}

function renderLogs(job) {
  const logs = job?.logs || [];
  $("logBox").textContent = logs
    .map((entry) => `[${entry.time}] ${entry.stage}: ${entry.message}`)
    .join("\n");
  $("logBox").scrollTop = $("logBox").scrollHeight;
}

function renderArtifacts(artifacts = {}) {
  setImage("whiteImage", artifacts.white_image_url);
  setImage("finalImage", artifacts.final_image_url);
  setImage("comparisonImage", artifacts.comparison_image_url);
  if (artifacts.model_path) {
    $("openWorkdir").href = `/api/file?path=${encodeURIComponent(artifacts.model_path)}`;
  }
}

function renderModelStatus(modelStatus) {
  const host = $("modelStatus");
  host.innerHTML = "";
  Object.entries(modelStatus || {}).forEach(([key, model]) => {
    const row = document.createElement("div");
    row.className = "model-row";
    const allPresent = (model.paths || []).every((item) => item.exists);
    row.innerHTML = `<strong title="${key}">${key}</strong><span class="${allPresent ? "" : "missing"}">${allPresent ? "ready" : "missing"}</span>`;
    host.appendChild(row);
  });
}

function applyModeVisibility() {
  const mode = $("sourceMode").value;
  document.querySelectorAll("[data-mode-field]").forEach((node) => {
    const modes = node.dataset.modeField.split(" ");
    node.style.display = modes.includes(mode) ? "grid" : "none";
  });
}

function syncSliderLabels() {
  $("cannyValue").textContent = Number($("cannyScale").value).toFixed(2);
  $("depthValue").textContent = Number($("depthScale").value).toFixed(2);
}

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  const defaults = state.config.defaults || {};
  $("prompt").value = defaults.prompt || "";
  $("negativePrompt").value = defaults.negative_prompt || "";
  $("renderResolution").value = defaults.render_resolution || 1536;
  $("aiWidth").value = defaults.ai_width || 1536;
  $("aiHeight").value = defaults.ai_height || 1536;
  $("steps").value = defaults.steps || 42;
  $("cannyScale").value = defaults.canny_scale || 2.85;
  $("depthScale").value = defaults.depth_scale || 0.55;
  syncSliderLabels();
  renderModelStatus(state.config.model_status);
  renderArtifacts(state.config.latest);
  setStatus(null);
}

async function uploadReferenceIfNeeded() {
  const file = $("referenceFile").files[0];
  if (!file) {
    return $("referencePath").value.trim();
  }
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "upload failed");
  }
  $("referencePath").value = data.path;
  return data.path;
}

function numberValue(id) {
  return Number($(id).value);
}

async function startRun(event) {
  event.preventDefault();
  $("runButton").disabled = true;
  const sourceMode = $("sourceMode").value;
  const payload = {
    source_mode: sourceMode,
    model_path: sourceMode === "model_path" || sourceMode === "existing_mesh" ? $("modelPath").value.trim() : "",
    prompt: $("prompt").value.trim(),
    negative_prompt: $("negativePrompt").value.trim(),
    render_resolution: numberValue("renderResolution"),
    ai_width: numberValue("aiWidth"),
    ai_height: numberValue("aiHeight"),
    candidates: numberValue("candidates"),
    steps: numberValue("steps"),
    seed: numberValue("seed"),
    canny_scale: numberValue("cannyScale"),
    depth_scale: numberValue("depthScale"),
    geometry_lock: $("geometryLock").checked,
  };
  if (sourceMode === "hunyuan_reference") {
    payload.reference_image = await uploadReferenceIfNeeded();
  }
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const job = await response.json();
  if (!response.ok) {
    throw new Error(job.error || "failed to start run");
  }
  state.activeJobId = job.id;
  setStatus(job);
  renderLogs(job);
  pollJob();
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }
  state.pollTimer = setInterval(pollJob, 2200);
}

async function pollJob() {
  if (!state.activeJobId) return;
  const response = await fetch(`/api/jobs/${state.activeJobId}`);
  const job = await response.json();
  setStatus(job);
  renderLogs(job);
  if (job.artifacts) {
    renderArtifacts(job.artifacts);
  }
  if (job.status === "complete" || job.status === "failed") {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    if (job.status === "failed") {
      $("logBox").textContent += `\nERROR: ${job.error || "failed"}`;
    }
  }
}

async function refreshAll() {
  await loadConfig();
  if (state.activeJobId) {
    await pollJob();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("sourceMode").addEventListener("change", applyModeVisibility);
  $("cannyScale").addEventListener("input", syncSliderLabels);
  $("depthScale").addEventListener("input", syncSliderLabels);
  $("runForm").addEventListener("submit", (event) => {
    startRun(event).catch((error) => {
      $("runButton").disabled = false;
      $("logBox").textContent += `\nERROR: ${error.message}`;
      $("statusPill").textContent = "failed";
      $("statusPill").className = "status-pill failed";
    });
  });
  $("refreshButton").addEventListener("click", () => refreshAll().catch(console.error));
  applyModeVisibility();
  loadConfig().catch((error) => {
    $("logBox").textContent = `ERROR: ${error.message}`;
  });
});
