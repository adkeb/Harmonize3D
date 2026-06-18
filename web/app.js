import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
import { GLTFLoader } from "/vendor/three/GLTFLoader.js";
import { OBJLoader } from "/vendor/three/OBJLoader.js";

const BASE_DISTANCE = 3.2;

const state = {
  config: null,
  modelPath: "",
  modelUrl: "",
  renderManifest: "",
  cameraLocked: false,
  cameraState: null,
  viewer: null,
  busyStage: "",
  artifacts: {},
  autoTaskId: "",
};

const $ = (id) => document.getElementById(id);

function setText(id, text) {
  $(id).textContent = text;
}

function setBusy(stage, busy) {
  state.busyStage = busy ? stage : "";
  document.querySelectorAll("button").forEach((button) => {
    if (button.id === "refreshButton") return;
    button.disabled = busy || button.dataset.disabled === "true";
  });
  syncButtonState();
}

function syncButtonState() {
  if (state.busyStage) return;
  $("lockCameraButton").disabled = !state.modelPath;
  $("renderWhiteButton").disabled = !(state.modelPath && state.cameraLocked);
  $("renderAiButton").disabled = !state.renderManifest;
}

function setStageStatus(id, text, status = "") {
  const node = $(id);
  node.textContent = text;
  node.className = status;
  renderStageSummary();
}

function appendArtifact(label, path, url) {
  if (!path && !url) return;
  state.artifacts[label] = { path, url };
  renderArtifactLinks();
}

function renderArtifactLinks() {
  $("artifactLinks").innerHTML = Object.entries(state.artifacts)
    .map(([label, artifact]) => {
      const href = artifact.url || `/api/file?path=${encodeURIComponent(artifact.path)}`;
      return `<a href="${href}" target="_blank" rel="noreferrer">${label}</a>`;
    })
    .join("");
}

function renderStageSummary() {
  const items = [
    ["Auto", $("autoStatus")?.textContent || "等待输入"],
    ["3D", $("sourceStatus").textContent],
    ["相机", $("cameraStatus").textContent],
    ["白模", $("whiteStatus").textContent],
    ["AI", $("aiStatus").textContent],
  ];
  $("stageSummary").innerHTML = items
    .map(([label, text]) => `<div><strong>${label}</strong><span>${text}</span></div>`)
    .join("");
}

function clearImage(id) {
  const image = $(id);
  const figure = image.closest("figure");
  image.removeAttribute("src");
  image.hidden = true;
  figure?.classList.add("is-empty");
}

function setImage(id, url) {
  const image = $(id);
  const figure = image.closest("figure");
  if (!url) {
    clearImage(id);
    return Promise.resolve();
  }
  const versionedUrl = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  image.hidden = false;
  figure?.classList.remove("is-empty");
  const loaded = new Promise((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error(`图片加载失败：${url}`));
  });
  image.src = versionedUrl;
  return image.decode ? image.decode().catch(() => loaded) : loaded;
}

async function apiJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || "request failed");
    error.data = data;
    throw error;
  }
  return data;
}

function setStageDetails(id, text = "") {
  const node = $(id);
  if (!node) return;
  node.textContent = text;
}

function formatApiError(error) {
  const data = error.data || {};
  const lines = [];
  if (data.error_type) lines.push(`类型：${data.error_type}`);
  if (data.returncode !== undefined) lines.push(`退出码：${data.returncode}`);
  if (data.memory) {
    const memory = data.memory;
    lines.push(
      `WSL 内存：可用 ${memory.mem_available_mib} MiB，Swap 可用 ${memory.swap_free_mib} MiB，合计 ${memory.available_plus_swap_mib} MiB`,
    );
  }
  if (data.reference_image) lines.push(`已保留参考图：${data.reference_image}`);
  if (data.workdir) lines.push(`工作目录：${data.workdir}`);
  if (data.next_action) lines.push(`建议：${data.next_action}`);
  return lines.join("\n");
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

function applyModeVisibility() {
  const mode = $("sourceMode").value;
  document.querySelectorAll("[data-mode-field]").forEach((node) => {
    const modes = node.dataset.modeField.split(" ");
    node.style.display = modes.includes(mode) ? "grid" : "none";
  });
}

function initViewer() {
  const host = $("viewer");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111214);

  const width = Math.max(1, host.clientWidth);
  const height = Math.max(1, host.clientHeight);
  const aspect = width / height;
  const camera = new THREE.OrthographicCamera(-1.35 * aspect, 1.35 * aspect, 1.35, -1.35, 0.01, 100);
  camera.position.set(1.7, 1.05, 2.4);
  camera.zoom = 1;
  camera.updateProjectionMatrix();

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0.05, 0);
  controls.minZoom = 0.35;
  controls.maxZoom = 6;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.7));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(-3, 5, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x9ed7ff, 1.2);
  fill.position.set(4, 2.5, 3);
  scene.add(fill);

  const grid = new THREE.GridHelper(3.5, 16, 0x334047, 0x23272b);
  grid.position.y = -0.92;
  scene.add(grid);

  state.viewer = { host, scene, camera, renderer, controls, object: null, grid };

  controls.addEventListener("change", () => {
    if (state.modelPath) {
      showCameraPreview();
    }
  });

  window.addEventListener("resize", resizeViewer);
  animateViewer();
}

function resizeViewer() {
  if (!state.viewer) return;
  const { host, camera, renderer } = state.viewer;
  const width = Math.max(1, host.clientWidth);
  const height = Math.max(1, host.clientHeight);
  const aspect = width / height;
  camera.left = -1.35 * aspect;
  camera.right = 1.35 * aspect;
  camera.top = 1.35;
  camera.bottom = -1.35;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function animateViewer() {
  requestAnimationFrame(animateViewer);
  if (!state.viewer) return;
  state.viewer.controls.update();
  state.viewer.renderer.render(state.viewer.scene, state.viewer.camera);
}

function clearModel() {
  if (!state.viewer?.object) return;
  state.viewer.scene.remove(state.viewer.object);
  state.viewer.object.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach((material) => material.dispose());
      } else {
        node.material.dispose();
      }
    }
  });
  state.viewer.object = null;
}

function normalizeModel(object, modelPath) {
  object.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxSize = Math.max(size.x, size.y, size.z) || 1;
  const scale = 1.8 / maxSize;
  object.scale.multiplyScalar(scale);
  object.position.copy(center).multiplyScalar(-scale);
  object.updateMatrixWorld(true);
  const normalizedBox = new THREE.Box3().setFromObject(object);
  if (state.viewer?.grid) {
    state.viewer.grid.position.y = normalizedBox.min.y - 0.02;
  }
  object.traverse((node) => {
    if (node.isMesh) {
      if (node.geometry?.computeVertexNormals) {
        node.geometry.computeVertexNormals();
      }
      node.castShadow = true;
      node.receiveShadow = true;
      node.material = new THREE.MeshStandardMaterial({
        color: 0xe9edf0,
        emissive: 0x222426,
        roughness: 0.46,
        metalness: 0.08,
        side: THREE.DoubleSide,
      });
    }
  });
  return object;
}

function loadModelIntoViewer(modelUrl, modelPath) {
  if (!state.viewer) initViewer();
  clearModel();
  $("viewerEmpty").style.display = "none";
  $("viewerPill").textContent = "loading";
  $("viewerPill").className = "status-pill running";
  const lower = modelPath.toLowerCase();
  const loader = lower.endsWith(".obj") ? new OBJLoader() : new GLTFLoader();
  loader.load(
    modelUrl,
    (asset) => {
      const object = asset.scene || asset;
      normalizeModel(object, modelPath);
      state.viewer.scene.add(object);
      state.viewer.object = object;
      setView("hero");
      $("viewerPill").textContent = "ready";
      $("viewerPill").className = "status-pill complete";
      setText("modelTitle", "3D 白模可交互预览");
      setText("activeModelPath", modelPath);
      setStageStatus("cameraStatus", "可拖转、缩放、平移，并固定当前视角", "complete");
      syncButtonState();
    },
    undefined,
    (error) => {
      $("viewerPill").textContent = "failed";
      $("viewerPill").className = "status-pill failed";
      setStageStatus("cameraStatus", `模型预览失败：${error.message || error}`, "failed");
    },
  );
}

function setView(name) {
  const presets = {
    front: { azimuth_deg: 0, elevation_deg: 12, distance_scale: 1.0 },
    side: { azimuth_deg: 90, elevation_deg: 12, distance_scale: 1.0 },
    back: { azimuth_deg: 180, elevation_deg: 12, distance_scale: 1.0 },
    hero: { azimuth_deg: 35, elevation_deg: 18, distance_scale: 1.0 },
  };
  const preset = presets[name] || presets.hero;
  applyCameraState({
    ...preset,
    ortho_scale: 2.7,
    target: [0, 0.05, 0],
    coordinate_space: "three_y_up",
  });
}

function applyCameraState(cameraState) {
  if (!state.viewer) return;
  const { camera, controls } = state.viewer;
  const azimuth = THREE.MathUtils.degToRad(cameraState.azimuth_deg);
  const elevation = THREE.MathUtils.degToRad(cameraState.elevation_deg);
  const distance = BASE_DISTANCE * cameraState.distance_scale;
  const horizontal = Math.cos(elevation) * distance;
  const target = new THREE.Vector3(...cameraState.target);
  camera.position.set(
    target.x + Math.sin(azimuth) * horizontal,
    target.y + Math.sin(elevation) * distance,
    target.z + Math.cos(azimuth) * horizontal,
  );
  controls.target.copy(target);
  camera.zoom = 2.7 / cameraState.ortho_scale;
  camera.updateProjectionMatrix();
  controls.update();
  showCameraPreview();
}

function currentCameraState() {
  const { host, camera, controls } = state.viewer;
  const target = controls.target.clone();
  const position = camera.position.clone();
  const offset = camera.position.clone().sub(target);
  const distance = Math.max(0.01, offset.length());
  const azimuth = THREE.MathUtils.radToDeg(Math.atan2(offset.x, offset.z));
  const elevation = THREE.MathUtils.radToDeg(Math.asin(offset.y / distance));
  const normalizedAzimuth = ((azimuth % 360) + 360) % 360;
  const viewportAspect = Math.max(0.2, Math.min(4, host.clientWidth / Math.max(host.clientHeight, 1)));
  return {
    azimuth_deg: Number(normalizedAzimuth.toFixed(3)),
    elevation_deg: Number(elevation.toFixed(3)),
    distance_scale: Number((distance / BASE_DISTANCE).toFixed(4)),
    ortho_scale: Number((2.7 / Math.max(camera.zoom, 0.001)).toFixed(4)),
    target: [Number(target.x.toFixed(4)), Number(target.y.toFixed(4)), Number(target.z.toFixed(4))],
    position: [Number(position.x.toFixed(4)), Number(position.y.toFixed(4)), Number(position.z.toFixed(4))],
    viewport_aspect: Number(viewportAspect.toFixed(4)),
    coordinate_space: "three_y_up",
  };
}

function showCameraPreview() {
  if (!state.viewer || !state.modelPath) return;
  $("cameraJson").textContent = JSON.stringify(currentCameraState(), null, 2);
}

function lockCamera() {
  if (!state.viewer || !state.modelPath) return;
  state.cameraState = currentCameraState();
  state.cameraLocked = true;
  $("cameraJson").textContent = JSON.stringify(state.cameraState, null, 2);
  setStageStatus("cameraStatus", "已固定当前视角，可渲染白模通道", "complete");
  syncButtonState();
}

async function fetchJson(path) {
  const response = await fetch(path);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || "request failed");
    error.data = data;
    throw error;
  }
  return data;
}

function autoLogText(job) {
  const stages = job.stages || [];
  const lines = stages
    .filter((stage) => stage.status !== "pending" || stage.logs?.length)
    .map((stage) => {
      const latest = stage.logs?.[stage.logs.length - 1]?.message || stage.message || "";
      const progress = Math.round((stage.progress || 0) * 100);
      return `${stage.label || stage.id}: ${stage.status} ${progress}% ${latest}`.trim();
    });
  if (job.error) lines.push(`error: ${job.error}`);
  return lines.join("\n");
}

async function applyAutoSummary(summary) {
  if (!summary) return;
  const urls = summary.artifact_urls || {};
  const artifacts = summary.artifacts || {};
  Object.entries(artifacts).forEach(([label, path]) => appendArtifact(label, path, urls[label]));
  if (summary.model_path) {
    state.modelPath = summary.model_path;
    setText("activeModelPath", summary.model_path);
    setText("modelTitle", "Auto Agent 3D 白模");
  }
  if (summary.render_manifest) state.renderManifest = summary.render_manifest;
  if (summary.final_image && (urls.final_image || summary.final_image_url)) {
    await setImage("finalImage", urls.final_image || summary.final_image_url);
  }
  if (summary.comparison_image && (urls.comparison_image || summary.comparison_image_url)) {
    await setImage("comparisonImage", urls.comparison_image || summary.comparison_image_url);
  }
  if (summary.render_manifest) appendArtifact("auto render manifest", summary.render_manifest, urls.render_manifest);
  if (summary.agent_report) {
    appendArtifact("auto agent report", summary.agent_report, urls.agent_report);
    $("reportLink").href = urls.agent_report || `/api/file?path=${encodeURIComponent(summary.agent_report)}`;
  }
}

async function pollAutoRun(taskId) {
  let job = null;
  for (;;) {
    job = await fetchJson(`/api/auto-run/${encodeURIComponent(taskId)}`);
    setStageStatus("autoStatus", `${job.stage || "running"} · ${Math.round((job.progress || 0) * 100)}%`, "running");
    setStageDetails("autoDetails", autoLogText(job));
    if (["done", "complete", "needs_review", "failed"].includes(job.status)) break;
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  if (job.status === "failed") {
    throw new Error(job.error || "Auto Agent failed");
  }
  await applyAutoSummary(job.summary);
  setStageStatus("autoStatus", job.status === "needs_review" ? "Auto Agent 完成，需要复核" : "Auto Agent 完成", "complete");
  return job;
}

async function applyAutoSceneSummary(summary = {}) {
  const urls = summary.artifact_urls || {};
  await applyAutoSummary(summary);
  if (summary.global_concept) appendArtifact("global concept", summary.global_concept, urls.global_concept);
  if (summary.scene_preview) appendArtifact("scene preview", summary.scene_preview, urls.scene_preview);
  if (summary.scene_model_path) appendArtifact("final scene glb", summary.scene_model_path, urls.scene_model_path);
  if (summary.scene_plan) appendArtifact("scene plan", summary.scene_plan, urls.scene_plan);
  if (summary.module_plan) appendArtifact("module plan", summary.module_plan, urls.module_plan);
  if (summary.module_asset_manifest) appendArtifact("module asset manifest", summary.module_asset_manifest, urls.module_asset_manifest);
  if (summary.module_scores) appendArtifact("module scores", summary.module_scores, urls.module_scores);
}

async function pollAutoScene(taskId) {
  let job = null;
  for (;;) {
    job = await fetchJson(`/api/auto-scene/${encodeURIComponent(taskId)}`);
    setStageStatus("autoStatus", `${job.stage || "running"} · ${Math.round((job.progress || 0) * 100)}%`, "running");
    setStageDetails("autoDetails", autoLogText(job));
    if (["done", "complete", "needs_review", "failed"].includes(job.status)) break;
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  if (job.status === "failed") {
    throw new Error(job.error || "Auto Scene failed");
  }
  await applyAutoSceneSummary(job.summary);
  setStageStatus("autoStatus", job.status === "needs_review" ? "Auto Scene 完成，需要复核" : "Auto Scene 完成", "complete");
  return job;
}

async function runAutoAgent() {
  setBusy("auto", true);
  setStageStatus("autoStatus", "Auto Agent 正在规划任务...", "running");
  setStageDetails("autoDetails");
  try {
    const result = await apiJson("/api/auto-run", {
      request: $("autoRequest").value.trim(),
      source_mode: $("autoSourceMode").value,
      model_path: $("autoSourceMode").value === "model_path" ? $("modelPath").value.trim() : "",
      reference_image: $("autoSourceMode").value === "image_to_3d" ? $("referencePath").value.trim() : "",
      output_views: Number($("autoViews").value),
      quality_mode: $("autoQuality").value,
      geometry_mode: $("autoGeometry").value,
      style_preset: $("autoStyle").value,
      num_candidates_per_view: numberValue("autoCandidates"),
      max_retries: 2,
      seed: numberValue("seed"),
      backend: $("autoDryRun").checked ? "mock" : "",
      dry_run: $("autoDryRun").checked,
    });
    state.autoTaskId = result.task_id;
    await pollAutoRun(result.task_id);
  } catch (error) {
    setStageStatus("autoStatus", `失败：${error.message}`, "failed");
    setStageDetails("autoDetails", formatApiError(error) || error.data?.traceback || "");
  } finally {
    setBusy("auto", false);
  }
}

async function runAutoScene() {
  setBusy("auto", true);
  setStageStatus("autoStatus", "Auto Scene 正在规划模块化场景...", "running");
  setStageDetails("autoDetails");
  try {
    const result = await apiJson("/api/auto-scene", {
      request: $("autoRequest").value.trim(),
      output_views: Number($("autoViews").value),
      quality_mode: $("autoQuality").value,
      geometry_mode: $("autoGeometry").value,
      style_preset: $("autoStyle").value === "clay_to_material" ? "exhibition" : $("autoStyle").value,
      num_candidates_per_view: numberValue("autoCandidates"),
      max_retries: 2,
      seed: numberValue("seed"),
      allow_procedural_fallback: true,
      render_backend: $("autoSceneRenderBackend").value,
      backend: $("autoDryRun").checked ? "mock" : "",
      dry_run: $("autoDryRun").checked,
    });
    state.autoTaskId = result.task_id;
    await pollAutoScene(result.task_id);
  } catch (error) {
    setStageStatus("autoStatus", `失败：${error.message}`, "failed");
    setStageDetails("autoDetails", formatApiError(error) || error.data?.traceback || "");
  } finally {
    setBusy("auto", false);
  }
}

async function generate3D() {
  setBusy("source", true);
  const mode = $("sourceMode").value;
  const message =
    mode === "prompt_3d"
      ? "正在调用 Flux2 Klein 生成参考图，然后调用 Hunyuan3D 2.1 生成白模..."
      : mode === "image_3d"
        ? "正在调用 Hunyuan3D 2.1 根据参考图生成白模..."
        : "正在载入 3D 白模...";
  setStageStatus("sourceStatus", message, "running");
  setStageDetails("sourceDetails");
  try {
    const payload = {
      source_mode: mode,
      prompt: $("sourcePrompt").value.trim(),
      model_path: mode === "model_path" ? $("modelPath").value.trim() : "",
      shape_quality: $("shapeQuality").value,
      seed: numberValue("seed"),
      reference_width: numberValue("referenceSize"),
      reference_height: numberValue("referenceSize"),
    };
    if (mode === "image_3d") {
      payload.reference_image = await uploadReferenceIfNeeded();
    }
    const result = await apiJson("/api/stage/3d", payload);
    state.modelPath = result.model_path;
    state.modelUrl = result.model_url;
    state.cameraLocked = false;
    state.cameraState = null;
    state.renderManifest = "";
    clearImage("whiteImage");
    clearImage("finalImage");
    clearImage("comparisonImage");
    setStageDetails("whiteDetails");
    setStageDetails("aiDetails");
    appendArtifact("3D 模型", result.model_path, result.model_url);
    if (result.metadata) appendArtifact("Hunyuan metadata", result.metadata, `/api/file?path=${encodeURIComponent(result.metadata)}`);
    if (result.reference_image_url) appendArtifact("参考图", result.reference_image, result.reference_image_url);
    const needsReview = result.status === "needs_review";
    setStageStatus("sourceStatus", needsReview ? "3D 白模已生成，但质量指标需要复核" : "3D 白模已就绪", needsReview ? "failed" : "complete");
    if (result.preprocessed_image_url) {
      appendArtifact("Hunyuan 预处理图", result.preprocessed_image, result.preprocessed_image_url);
    }
    if (result.hunyuan_profile) {
      const sanity = result.mesh_sanity ? `\nMesh sanity：${JSON.stringify(result.mesh_sanity, null, 2)}` : "";
      setStageDetails(
        "sourceDetails",
        `Hunyuan profile：${result.hunyuan_profile}\n参数：${JSON.stringify(result.hunyuan_parameters || {}, null, 2)}${sanity}`,
      );
    }
    setStageStatus("whiteStatus", "等待固定视角", "");
    setStageStatus("aiStatus", "等待白模通道", "");
    loadModelIntoViewer(result.model_url, result.model_path);
  } catch (error) {
    if (error.data?.reference_image_url) {
      appendArtifact("失败前参考图", error.data.reference_image, error.data.reference_image_url);
    }
    setStageStatus("sourceStatus", `失败：${error.message}`, "failed");
    setStageDetails("sourceDetails", formatApiError(error));
  } finally {
    setBusy("source", false);
  }
}

async function renderWhite() {
  if (!state.modelPath || !state.cameraState) return;
  setBusy("white", true);
  setStageStatus("whiteStatus", "Blender 正在按固定视角渲染通道...", "running");
  setStageDetails("whiteDetails");
  try {
    state.cameraState = currentCameraState();
    $("cameraJson").textContent = JSON.stringify(state.cameraState, null, 2);
    const result = await apiJson("/api/stage/white-render", {
      model_path: state.modelPath,
      camera: state.cameraState,
      resolution: numberValue("renderResolution"),
      samples: numberValue("renderSamples"),
    });
    state.renderManifest = result.render_manifest;
    await setImage("whiteImage", result.white_image_url);
    appendArtifact("白模 manifest", result.render_manifest, result.render_manifest_url);
    appendArtifact("固定视角白模", result.white_image, result.white_image_url);
    setStageStatus("whiteStatus", "白模通道已生成", "complete");
    setStageStatus("aiStatus", "可输入提示词生成最终渲染", "complete");
  } catch (error) {
    setStageStatus("whiteStatus", `失败：${error.message}`, "failed");
    setStageDetails("whiteDetails", formatApiError(error));
  } finally {
    setBusy("white", false);
  }
}

async function renderAI() {
  if (!state.renderManifest) return;
  setBusy("ai", true);
  setStageStatus("aiStatus", "AI 正在根据固定视角白模生成最终图...", "running");
  setStageDetails("aiDetails");
  try {
    const backendMode = $("aiBackend").value;
    const result = await apiJson("/api/stage/ai-render", {
      render_manifest: state.renderManifest,
      prompt: $("renderPrompt").value.trim(),
      negative_prompt: $("negativePrompt").value.trim(),
      agent_render: backendMode === "agent",
      backend: backendMode === "mock" ? "mock" : "",
      model_key: "flux2_klein_4b",
      max_generations: numberValue("agentBudget"),
      steps: numberValue("aiSteps"),
      width: numberValue("aiResolution"),
      height: numberValue("aiResolution"),
      seed: numberValue("seed"),
      expand_views: true,
    });
    if (!result.final_image_url || !result.comparison_image_url) {
      throw new Error("AI 渲染完成但缺少最终图或对照图 URL");
    }
    await Promise.all([setImage("finalImage", result.final_image_url), setImage("comparisonImage", result.comparison_image_url)]);
    appendArtifact("最终渲染", result.final_image, result.final_image_url);
    appendArtifact("对照图", result.comparison_image, result.comparison_image_url);
    if (result.agent_report_url) {
      appendArtifact("Agent report", result.agent_report, result.agent_report_url);
      $("reportLink").href = result.agent_report_url;
      if (result.multiview_contact_sheet_url) {
        appendArtifact("多视角 contact sheet", result.multiview_contact_sheet, result.multiview_contact_sheet_url);
      }
    } else if (result.score_report) {
      appendArtifact("评分 report", result.score_report, `/api/file?path=${encodeURIComponent(result.score_report)}`);
      $("reportLink").href = `/api/file?path=${encodeURIComponent(result.score_report)}`;
    }
    setStageStatus("aiStatus", result.status === "needs_review" ? "最终渲染已完成，Agent 标记需要复核" : "最终渲染已完成", "complete");
  } catch (error) {
    setStageStatus("aiStatus", `失败：${error.message}`, "failed");
    setStageDetails("aiDetails", formatApiError(error));
  } finally {
    setBusy("ai", false);
  }
}

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  const defaults = state.config.defaults || {};
  $("renderResolution").value = defaults.render_resolution || 1024;
  $("renderPrompt").value = defaults.prompt || $("renderPrompt").value;
  $("negativePrompt").value = defaults.negative_prompt || $("negativePrompt").value;
  $("shapeQuality").value = defaults.shape_quality || "balanced";
  $("aiBackend").value = defaults.agent_render === false ? "flux2" : "agent";
  $("agentBudget").value = String(defaults.agent_max_generations || 10);
  $("aiSteps").value = String(defaults.ai_steps || 12);
  $("aiResolution").value = String(defaults.ai_resolution || 1024);
  renderStageSummary();
}

document.addEventListener("DOMContentLoaded", () => {
  initViewer();
  applyModeVisibility();
  $("sourceMode").addEventListener("change", applyModeVisibility);
  $("autoRunButton").addEventListener("click", runAutoAgent);
  $("autoSceneButton").addEventListener("click", runAutoScene);
  $("generate3dButton").addEventListener("click", generate3D);
  $("lockCameraButton").addEventListener("click", lockCamera);
  $("renderWhiteButton").addEventListener("click", renderWhite);
  $("renderAiButton").addEventListener("click", renderAI);
  $("resetCameraButton").addEventListener("click", () => setView("hero"));
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $("refreshButton").addEventListener("click", () => loadConfig().catch(console.error));
  loadConfig().catch((error) => {
    setStageStatus("sourceStatus", `配置读取失败：${error.message}`, "failed");
  });
  syncButtonState();
});
