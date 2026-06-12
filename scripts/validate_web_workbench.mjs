import { chromium } from "playwright";

const baseUrl = process.env.H3D_VALIDATE_BASE_URL || "http://127.0.0.1:7866";
const sourceMode = process.env.H3D_VALIDATE_SOURCE_MODE || "model_path";
const modelPath = process.env.H3D_VALIDATE_MODEL_PATH || "/root/sakura/work/Harmonize3D/examples/sample_model.obj";
const referenceImage = process.env.H3D_VALIDATE_REFERENCE_IMAGE || "";
const sourcePrompt =
  process.env.H3D_VALIDATE_SOURCE_PROMPT ||
  "high quality white sports car concept, clean side view, strong silhouette, centered product reference";
const renderPrompt =
  process.env.H3D_VALIDATE_RENDER_PROMPT ||
  "same exact 3D model silhouette, premium white and graphite hypercar material, black glass canopy, studio automotive product render, crisp reflections, preserve wheel arches, rear wing, vents and proportions";
const negativePrompt =
  process.env.H3D_VALIDATE_NEGATIVE_PROMPT ||
  "changed silhouette, missing wheels, extra wheels, extra spoiler, deformed car, noisy texture, rough gritty surface, text, watermark, logo";
const aiBackend = process.env.H3D_VALIDATE_AI_BACKEND || "mock";
const shapeQuality = process.env.H3D_VALIDATE_SHAPE_QUALITY || "balanced";
const screenshotPath = process.env.H3D_VALIDATE_SCREENSHOT || "/tmp/harmonize3d-final-validation.png";
const timeoutMs = Number(process.env.H3D_VALIDATE_TIMEOUT_MS || "420000");

async function setSelect(page, selector, value) {
  await page.selectOption(selector, value);
  await page.dispatchEvent(selector, "change");
}

async function setValue(page, selector, value) {
  await page.fill(selector, String(value));
  await page.dispatchEvent(selector, "input");
  await page.dispatchEvent(selector, "change");
}

async function waitForStatus(page, selector, text) {
  await page.waitForFunction(
    ({ selector, text }) => document.querySelector(selector)?.textContent?.includes(text),
    { selector, text },
    { timeout: timeoutMs },
  );
}

async function assertVisibleImage(page, id) {
  await page.waitForFunction(
    (id) => {
      const image = document.getElementById(id);
      return image && !image.hidden && image.complete && image.naturalWidth > 0 && image.naturalHeight > 0;
    },
    id,
    { timeout: timeoutMs },
  );
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  page.setDefaultTimeout(timeoutMs);
  await page.goto(baseUrl, { waitUntil: "networkidle" });

  await setSelect(page, "#sourceMode", sourceMode);
  await setSelect(page, "#shapeQuality", shapeQuality);
  await setValue(page, "#renderResolution", process.env.H3D_VALIDATE_RENDER_RESOLUTION || "512");
  await setValue(page, "#renderSamples", process.env.H3D_VALIDATE_RENDER_SAMPLES || "16");
  await setSelect(page, "#aiBackend", aiBackend);
  await setSelect(page, "#agentBudget", process.env.H3D_VALIDATE_AGENT_BUDGET || "1");
  await setValue(page, "#aiSteps", process.env.H3D_VALIDATE_AI_STEPS || "8");
  await setSelect(page, "#aiResolution", process.env.H3D_VALIDATE_AI_RESOLUTION || "1024");
  await setValue(page, "#renderPrompt", renderPrompt);
  await setValue(page, "#negativePrompt", negativePrompt);

  if (sourceMode === "model_path") {
    await setValue(page, "#modelPath", modelPath);
  } else if (sourceMode === "image_3d") {
    if (!referenceImage) throw new Error("H3D_VALIDATE_REFERENCE_IMAGE is required for image_3d validation.");
    await setValue(page, "#referencePath", referenceImage);
  } else {
    await setValue(page, "#sourcePrompt", sourcePrompt);
  }

  await page.click("#generate3dButton");
  await waitForStatus(page, "#sourceStatus", "3D 白模");
  await page.waitForFunction(() => document.querySelector("#viewer canvas")?.width > 0, null, { timeout: 60000 });

  await page.click("#lockCameraButton");
  await waitForStatus(page, "#cameraStatus", "已固定");

  await page.click("#renderWhiteButton");
  await waitForStatus(page, "#whiteStatus", "白模通道已生成");
  await assertVisibleImage(page, "whiteImage");

  await page.click("#renderAiButton");
  await waitForStatus(page, "#aiStatus", "最终渲染已完成");
  await assertVisibleImage(page, "finalImage");
  await assertVisibleImage(page, "comparisonImage");

  const broken = await page.$$eval("img:not([hidden])", (images) =>
    images.filter((image) => !image.complete || image.naturalWidth === 0).map((image) => image.id),
  );
  if (broken.length) {
    throw new Error(`Broken visible images: ${broken.join(", ")}`);
  }

  await page.screenshot({ path: screenshotPath, fullPage: true });
  await browser.close();
  console.log(JSON.stringify({ status: "complete", screenshot: screenshotPath }, null, 2));
}

main().catch(async (error) => {
  console.error(error);
  process.exit(1);
});
