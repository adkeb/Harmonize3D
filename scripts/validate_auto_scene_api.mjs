const baseUrl = process.env.H3D_VALIDATE_BASE_URL || "http://127.0.0.1:7866";
const timeoutMs = Number(process.env.H3D_VALIDATE_TIMEOUT_MS || "240000");
const pollMs = Number(process.env.H3D_VALIDATE_POLL_MS || "1000");

const requiredStages = [
  "understand",
  "concept",
  "decompose",
  "module_reference",
  "module_3d",
  "module_check",
  "layout",
  "scene_preview",
  "camera",
  "render",
  "agent",
  "score",
  "consistency",
  "package",
  "complete",
];

async function apiJson(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (error) {
    throw new Error(`Non-JSON response from ${path}: ${text.slice(0, 240)}`);
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${path}: ${JSON.stringify(data)}`);
  }
  return data;
}

async function main() {
  const started = Date.now();
  const launched = await apiJson("/api/auto-scene", {
    method: "POST",
    body: JSON.stringify({
      request:
        process.env.H3D_VALIDATE_AUTO_SCENE_REQUEST ||
        "生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。",
      output_views: Number(process.env.H3D_VALIDATE_AUTO_SCENE_VIEWS || "3"),
      quality_mode: process.env.H3D_VALIDATE_AUTO_SCENE_QUALITY || "fast",
      geometry_mode: process.env.H3D_VALIDATE_AUTO_SCENE_GEOMETRY || "strict",
      style_preset: process.env.H3D_VALIDATE_AUTO_SCENE_STYLE || "exhibition",
      num_candidates_per_view: Number(process.env.H3D_VALIDATE_AUTO_SCENE_CANDIDATES || "1"),
      max_retries: Number(process.env.H3D_VALIDATE_AUTO_SCENE_RETRIES || "0"),
      allow_procedural_fallback: true,
      render_backend: process.env.H3D_VALIDATE_AUTO_SCENE_RENDER_BACKEND || "auto",
      backend: "mock",
      dry_run: true,
      no_llm: true,
    }),
  });

  let job = launched.job || launched;
  while (!["done", "complete", "needs_review", "failed"].includes(job.status)) {
    if (Date.now() - started > timeoutMs) {
      throw new Error(`Timed out waiting for Auto Scene job ${launched.task_id}`);
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
    job = await apiJson(`/api/auto-scene/${encodeURIComponent(launched.task_id)}`);
  }
  if (job.status === "failed") {
    throw new Error(job.error || `Auto Scene failed: ${JSON.stringify(job)}`);
  }

  const stages = new Map((job.stages || []).map((stage) => [stage.id, stage]));
  const missingStages = requiredStages.filter((stage) => !stages.has(stage));
  if (missingStages.length) {
    throw new Error(`Missing Auto Scene stages: ${missingStages.join(", ")}`);
  }
  const missingRetryCount = requiredStages.filter((stage) => !Object.hasOwn(stages.get(stage), "retry_count"));
  if (missingRetryCount.length) {
    throw new Error(`Stages missing retry_count: ${missingRetryCount.join(", ")}`);
  }
  const missingWarnings = requiredStages.filter((stage) => !Object.hasOwn(stages.get(stage), "warnings"));
  if (missingWarnings.length) {
    throw new Error(`Stages missing warnings: ${missingWarnings.join(", ")}`);
  }
  const missingError = requiredStages.filter((stage) => !Object.hasOwn(stages.get(stage), "error"));
  if (missingError.length) {
    throw new Error(`Stages missing error: ${missingError.join(", ")}`);
  }

  const summary = job.summary || {};
  const requiredArtifacts = [
    "auto_task",
    "scene_plan",
    "concept_image_plan",
    "global_concept",
    "module_plan",
    "module_asset_manifest",
    "scene_assembly",
    "final_scene_manifest",
    "scene_model_path",
    "scene_preview",
    "camera_plan",
    "render_manifest",
    "module_scores",
    "agent_report",
    "tool_calls",
    "visual_judgement",
    "final_image",
    "comparison_image",
    "contact_sheet",
  ];
  const missingArtifacts = requiredArtifacts.filter((key) => !summary[key]);
  if (missingArtifacts.length) {
    throw new Error(`Summary missing artifacts: ${missingArtifacts.join(", ")}`);
  }

  console.log(
    JSON.stringify(
      {
        status: "complete",
        task_id: launched.task_id,
        job_status: job.status,
        workdir: summary.workdir,
        stages: requiredStages,
        artifacts: requiredArtifacts,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
