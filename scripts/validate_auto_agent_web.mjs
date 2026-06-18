import { chromium } from "playwright";

const baseUrl = process.env.H3D_VALIDATE_BASE_URL || "http://127.0.0.1:7866";
const requestText =
  process.env.H3D_VALIDATE_AUTO_REQUEST ||
  "生成一辆未来感白色电动跑车，放在灰色摄影棚中，从低角度三分之四前视角渲染三张产品图。";
const screenshotPath = process.env.H3D_VALIDATE_AUTO_SCREENSHOT || "/tmp/harmonize3d-auto-agent-validation.png";
const timeoutMs = Number(process.env.H3D_VALIDATE_TIMEOUT_MS || "240000");

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

  await setValue(page, "#autoRequest", requestText);
  await setSelect(page, "#autoSourceMode", process.env.H3D_VALIDATE_AUTO_SOURCE_MODE || "procedural");
  await setSelect(page, "#autoViews", process.env.H3D_VALIDATE_AUTO_VIEWS || "3");
  await setSelect(page, "#autoQuality", process.env.H3D_VALIDATE_AUTO_QUALITY || "fast");
  await setSelect(page, "#autoGeometry", process.env.H3D_VALIDATE_AUTO_GEOMETRY || "strict");
  await setSelect(page, "#autoStyle", process.env.H3D_VALIDATE_AUTO_STYLE || "product");
  await setValue(page, "#autoCandidates", process.env.H3D_VALIDATE_AUTO_CANDIDATES || "1");
  await page.setChecked("#autoDryRun", true);

  await page.click("#autoRunButton");
  await waitForStatus(page, "#autoStatus", "Auto Agent 完成");
  await assertVisibleImage(page, "finalImage");
  await assertVisibleImage(page, "comparisonImage");

  const details = await page.textContent("#autoDetails");
  const requiredStages = ["理解需求", "扩写提示", "规划流程", "模型检查", "白模通道", "候选评分", "打包产物"];
  const missing = requiredStages.filter((stage) => !details?.includes(stage));
  if (missing.length) {
    throw new Error(`Auto Agent details missing stages: ${missing.join(", ")}`);
  }

  await page.screenshot({ path: screenshotPath, fullPage: true });
  await browser.close();
  console.log(JSON.stringify({ status: "complete", screenshot: screenshotPath }, null, 2));
}

main().catch(async (error) => {
  console.error(error);
  process.exit(1);
});
