const PROCESS_ATTR = "data-mangashift-state";
const OVERLAY_ATTR = "data-mangashift-overlay";
const DB_NAME = "mangashift_cache_v1";
const STORE_NAME = "processed_panels";
const MIN_W = 200;
const MIN_H = 150;

const DEFAULT_SETTINGS = {
  enabled: true,
  autoEnhance: true,
  sourceLang: "auto",
  style: "smooth",
  colorize: false,
  upscale: 1.0,
  rememberSeriesStyle: true,
};

let effectiveSettings = { ...DEFAULT_SETTINGS };

function installOverlayStyles() {
  if (document.getElementById("mangashift-style")) return;
  const style = document.createElement("style");
  style.id = "mangashift-style";
  style.textContent = `
    @keyframes mangashift-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .mangashift-chip {
      position: absolute; z-index: 2147483645; top: 8px; right: 8px;
      background: rgba(10,10,12,.86); color: #fff; border: 1px solid rgba(255,255,255,.25);
      border-radius: 999px; padding: 4px 8px; font-size: 11px; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      display: inline-flex; align-items: center; gap: 6px;
      pointer-events: auto;
    }
    .mangashift-spin {
      width: 10px; height: 10px; border-radius: 50%;
      border: 2px solid rgba(255,255,255,.3); border-top-color: #58d3ff;
      animation: mangashift-spin 1s linear infinite;
    }
    .mangashift-retry {
      border: 1px solid rgba(255,255,255,.35); background: rgba(255,255,255,.1);
      color: #fff; border-radius: 10px; font-size: 10px; padding: 2px 6px; cursor: pointer;
    }
  `;
  document.documentElement.appendChild(style);
}

function pageDomain() {
  return location.hostname.replace(/^www\./, "");
}

function seriesKeyFromUrl() {
  const parts = location.pathname.split("/").filter(Boolean);
  return parts.slice(0, 3).join("/") || "default";
}

function sha256Text(text) {
  const data = new TextEncoder().encode(text);
  return crypto.subtle.digest("SHA-256", data).then((hash) => {
    const bytes = new Uint8Array(hash);
    return [...bytes].map((x) => x.toString(16).padStart(2, "0")).join("");
  });
}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "key" });
        store.createIndex("ts", "ts");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function cacheGet(key) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = () => resolve(req.result?.value || null);
    req.onerror = () => reject(req.error);
  });
}

async function cacheSet(key, value) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put({ key, value, ts: Date.now() });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function isLikelyPanelImage(img) {
  if (!(img instanceof HTMLImageElement)) return false;
  if (!img.src || img.src.startsWith("data:")) return false;
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  if (w < MIN_W || h < MIN_H) return false;
  const ratio = w / h;
  if (ratio < 0.3 || ratio > 3.0) return false;
  const skip = img.closest("header, nav, footer, .avatar, .logo, .icon, .thumbnail, .thumb");
  return !skip;
}

function makeOverlay(img, text, withSpinner = true, withRetry = false, onRetry = null) {
  clearOverlay(img);
  const parent = img.parentElement;
  if (!parent) return;
  const parentStyle = window.getComputedStyle(parent);
  if (parentStyle.position === "static") parent.style.position = "relative";
  const chip = document.createElement("div");
  chip.className = "mangashift-chip";
  chip.setAttribute(OVERLAY_ATTR, "1");
  if (withSpinner) {
    const spin = document.createElement("span");
    spin.className = "mangashift-spin";
    chip.appendChild(spin);
  }
  const label = document.createElement("span");
  label.textContent = text;
  chip.appendChild(label);
  if (withRetry && onRetry) {
    const btn = document.createElement("button");
    btn.className = "mangashift-retry";
    btn.textContent = "Retry";
    btn.onclick = (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      onRetry();
    };
    chip.appendChild(btn);
  }
  parent.appendChild(chip);
}

function clearOverlay(img) {
  const parent = img.parentElement;
  if (!parent) return;
  parent.querySelectorAll(`[${OVERLAY_ATTR}]`).forEach((n) => n.remove());
}

function sendProcessMessage(payload) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type: "PROCESS_IMAGE", payload }, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!response || !response.ok) {
        reject(new Error(response?.error || "Unknown process failure"));
        return;
      }
      resolve(response);
    });
  });
}

async function processImage(img) {
  if (!effectiveSettings.enabled || !effectiveSettings.autoEnhance) return;
  if (!isLikelyPanelImage(img)) return;
  if (img.getAttribute(PROCESS_ATTR) === "done" || img.getAttribute(PROCESS_ATTR) === "pending") return;

  img.setAttribute(PROCESS_ATTR, "pending");
  img.style.opacity = "0.8";
  makeOverlay(img, "Enhancing", true, false);

  const src = img.currentSrc || img.src;
  const cacheKey = await sha256Text(
    `${src}|${effectiveSettings.sourceLang}|${effectiveSettings.style}|${effectiveSettings.upscale}|${effectiveSettings.colorize}`
  );

  try {
    const cached = await cacheGet(cacheKey);
    if (cached) {
      img.src = cached;
      img.style.opacity = "1";
      img.setAttribute(PROCESS_ATTR, "done");
      clearOverlay(img);
      return;
    }

    const response = await sendProcessMessage({
      imageUrl: src,
      filename: `panel_${Date.now()}.png`,
      sourceLang: effectiveSettings.sourceLang,
      targetLang: "en",
      style: effectiveSettings.style,
      colorize: effectiveSettings.colorize,
      upscale: effectiveSettings.upscale,
      preferredStyle: effectiveSettings.preferredStyle || "",
    });
    const dataUrl = `data:${response.contentType};base64,${response.imageBase64}`;
    img.src = dataUrl;
    await cacheSet(cacheKey, dataUrl);
    img.style.opacity = "1";
    img.setAttribute(PROCESS_ATTR, "done");
    clearOverlay(img);
  } catch (error) {
    img.style.opacity = "1";
    img.setAttribute(PROCESS_ATTR, "error");
    makeOverlay(img, "Enhance failed", false, true, () => {
      img.removeAttribute(PROCESS_ATTR);
      processImage(img);
    });
    console.warn("[MangaShift] process failed", src, error);
  }
}

function scanAndProcess(root = document) {
  root.querySelectorAll("img").forEach((img) => {
    if (img.complete) {
      processImage(img);
    } else {
      img.addEventListener("load", () => processImage(img), { once: true });
    }
  });
}

function observeDom() {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      mutation.addedNodes.forEach((node) => {
        if (!(node instanceof Element)) return;
        if (node.tagName === "IMG") {
          const img = node;
          if (img.complete) processImage(img);
          else img.addEventListener("load", () => processImage(img), { once: true });
          return;
        }
        scanAndProcess(node);
      });
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

function loadEffectiveSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["mangashiftSettings", "mangashiftSiteSettings", "mangashiftSeriesStyles"], (data) => {
      const global = { ...DEFAULT_SETTINGS, ...(data.mangashiftSettings || {}) };
      const perSiteMap = data.mangashiftSiteSettings || {};
      const seriesMap = data.mangashiftSeriesStyles || {};
      const site = perSiteMap[pageDomain()] || {};
      const key = `${pageDomain()}:${seriesKeyFromUrl()}`;
      const remembered = seriesMap[key] || "";
      effectiveSettings = {
        ...global,
        ...site,
        preferredStyle: global.rememberSeriesStyle ? remembered : "",
      };
      resolve(effectiveSettings);
    });
  });
}

async function init() {
  installOverlayStyles();
  await loadEffectiveSettings();
  if (!effectiveSettings.enabled || !effectiveSettings.autoEnhance) return;
  scanAndProcess(document);
  observeDom();
}

init().catch((err) => console.error("[MangaShift] init failed", err));
