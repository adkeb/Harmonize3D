const baseUrl = process.env.H3D_VALIDATE_BASE_URL || "http://127.0.0.1:7866";
const timeoutMs = Number(process.env.H3D_VALIDATE_TIMEOUT_MS || "240000");
const pollMs = Number(process.env.H3D_VALIDATE_POLL_MS || "1000");

const requiredStages = [
  "understand",
  "expand",
  "plan",
  "source",
  "mesh_check",
  "camera",
  "render",
  "agent",
  "score",
  "retry",
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
  const launched = await apiJson("/api/auto-run", {
    method: "POST",
    body: JSON.stringify({
      request:
        process.env.H3D_VALIDATE_AUTO_REQUEST ||
        "生成一辆未来感白色电动跑车，放在灰色摄影棚中，从低角度三分之四前视角渲染三张产品图。",
      source_mode: process.env.H3D_VALIDATE_AUTO_SOURCE_MODE || "procedural",
      output_views: Number(process.env.H3D_VALIDATE_AUTO_VIEWS || "3"),
      quality_mode: process.env.H3D_VALIDATE_AUTO_QUALITY || "fast",
      geometry_mode: process.env.H3D_VALIDATE_AUTO_GEOMETRY || "strict",
      style_preset: process.env.H3D_VALIDATE_AUTO_STYLE || "product",
      num_candidates_per_view: Number(process.env.H3D_VALIDATE_AUTO_CANDIDATES || "1"),
      max_retries: Number(process.env.H3D_VALIDATE_AUTO_RETRIES || "0"),
      backend: "mock",
      dry_run: true,
      no_llm: true,
    }),
  });

  let job = launched.job || launched;
  while (!["done", "complete", "needs_review", "failed"].includes(job.status)) {
    if (Date.now() - started > timeoutMs) {
      throw new Error(`Timed out waiting for Auto Agent job ${launched.task_id}`);
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
    job = await apiJson(`/api/auto-run/${encodeURIComponent(launched.task_id)}`);
  }
  if (job.status === "failed") {
    throw new Error(job.error || `Auto Agent failed: ${JSON.stringify(job)}`);
  }

  const stages = new Map((job.stages || []).map((stage) => [stage.id, stage]));
  const missingStages = requiredStages.filter((stage) => !stages.has(stage));
  if (missingStages.length) {
    throw new Error(`Missing Auto Agent stages: ${missingStages.join(", ")}`);
  }
  const missingRetryCount = requiredStages.filter((stage) => !Object.hasOwn(stages.get(stage), "retry_count"));
  if (missingRetryCount.length) {
    throw new Error(`Stages missing retry_count: ${missingRetryCount.join(", ")}`);
  }

  const summary = job.summary || {};
  const requiredArtifacts = ["auto_task", "prompt_plan", "camera_plan", "mesh_sanity", "render_manifest", "agent_report", "tool_calls", "visual_judgement", "final_image", "comparison_image", "contact_sheet"];
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
