const DEFAULT_SETTINGS = {
  enabled: true,
  autoEnhance: true,
  sourceLang: "auto",
  style: "smooth",
  colorize: false,
  upscale: 1.0,
  rememberSeriesStyle: true,
};

let activeTab = null;
let styles = [];
let domain = "unknown";
let seriesKey = "default";

const els = {
  serverState: document.getElementById("serverState"),
  enabledToggle: document.getElementById("enabledToggle"),
  autoToggle: document.getElementById("autoToggle"),
  rememberToggle: document.getElementById("rememberToggle"),
  sourceLang: document.getElementById("sourceLang"),
  colorizeToggle: document.getElementById("colorizeToggle"),
  upscaleRange: document.getElementById("upscaleRange"),
  upscaleValue: document.getElementById("upscaleValue"),
  styleGrid: document.getElementById("styleGrid"),
  applyBtn: document.getElementById("applyBtn"),
  clearSeriesBtn: document.getElementById("clearSeriesBtn"),
};

function getStorage(keys) {
  return new Promise((resolve) => chrome.storage.sync.get(keys, resolve));
}

function setStorage(payload) {
  return new Promise((resolve) => chrome.storage.sync.set(payload, resolve));
}

function parseDomainAndSeries(urlString) {
  try {
    const u = new URL(urlString);
    const host = u.hostname.replace(/^www\./, "");
    const parts = u.pathname.split("/").filter(Boolean);
    return { domain: host, seriesKey: parts.slice(0, 3).join("/") || "default" };
  } catch {
    return { domain: "unknown", seriesKey: "default" };
  }
}

async function loadStyles() {
  const response = await fetch(chrome.runtime.getURL("style_presets.json"));
  styles = await response.json();
}

function updateServerBadge(ok, text) {
  els.serverState.className = `badge ${ok ? "ok" : "error"}`;
  els.serverState.textContent = text;
}

async function checkServer() {
  try {
    const r = await fetch("http://127.0.0.1:8000/health");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    updateServerBadge(true, "Connected");
  } catch {
    updateServerBadge(false, "Offline");
  }
}

function renderStyleGrid(selectedStyle, onSelect) {
  els.styleGrid.innerHTML = "";
  styles.forEach((preset) => {
    const btn = document.createElement("button");
    btn.className = `style-card ${preset.id === selectedStyle ? "selected" : ""}`;
    btn.type = "button";
    btn.innerHTML = `
      <img src="${preset.thumbnail_path}" alt="${preset.name}" />
      <div class="name">${preset.name}</div>
      <div class="hint">${preset.prompt_hint}</div>
    `;
    btn.addEventListener("click", () => onSelect(preset.id));
    els.styleGrid.appendChild(btn);
  });
}

async function init() {
  [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const parsed = parseDomainAndSeries(activeTab?.url || "");
  domain = parsed.domain;
  seriesKey = parsed.seriesKey;

  await loadStyles();
  await checkServer();

  const data = await getStorage(["mangashiftSettings", "mangashiftSiteSettings", "mangashiftSeriesStyles"]);
  const global = { ...DEFAULT_SETTINGS, ...(data.mangashiftSettings || {}) };
  const siteMap = data.mangashiftSiteSettings || {};
  const seriesMap = data.mangashiftSeriesStyles || {};
  const site = siteMap[domain] || {};
  const seriesStorageKey = `${domain}:${seriesKey}`;
  const rememberedStyle = seriesMap[seriesStorageKey] || "";

  const state = {
    ...global,
    ...site,
    style: rememberedStyle || site.style || global.style,
  };

  els.enabledToggle.checked = Boolean(state.enabled);
  els.autoToggle.checked = Boolean(state.autoEnhance);
  els.rememberToggle.checked = Boolean(state.rememberSeriesStyle);
  els.sourceLang.value = state.sourceLang;
  els.colorizeToggle.checked = Boolean(state.colorize);
  els.upscaleRange.value = String(state.upscale);
  els.upscaleValue.textContent = String(state.upscale);

  const persist = async (patch = {}) => {
    const nextGlobal = { ...global, ...patch };
    const nextSiteMap = { ...siteMap, [domain]: { ...site, ...patch } };
    const payload = {
      mangashiftSettings: nextGlobal,
      mangashiftSiteSettings: nextSiteMap,
    };
    if (nextGlobal.rememberSeriesStyle) {
      payload.mangashiftSeriesStyles = { ...seriesMap, [seriesStorageKey]: nextGlobal.style };
    }
    await setStorage(payload);
  };

  const selectStyle = async (styleId) => {
    global.style = styleId;
    renderStyleGrid(styleId, selectStyle);
    await persist({ style: styleId });
    if (els.rememberToggle.checked) {
      await setStorage({
        mangashiftSeriesStyles: { ...seriesMap, [seriesStorageKey]: styleId },
      });
    }
  };

  renderStyleGrid(state.style, selectStyle);

  els.enabledToggle.addEventListener("change", async () => {
    await persist({ enabled: els.enabledToggle.checked });
  });
  els.autoToggle.addEventListener("change", async () => {
    await persist({ autoEnhance: els.autoToggle.checked });
  });
  els.rememberToggle.addEventListener("change", async () => {
    await persist({ rememberSeriesStyle: els.rememberToggle.checked });
  });
  els.sourceLang.addEventListener("change", async () => {
    await persist({ sourceLang: els.sourceLang.value });
  });
  els.colorizeToggle.addEventListener("change", async () => {
    await persist({ colorize: els.colorizeToggle.checked });
  });
  els.upscaleRange.addEventListener("input", () => {
    els.upscaleValue.textContent = els.upscaleRange.value;
  });
  els.upscaleRange.addEventListener("change", async () => {
    await persist({ upscale: Number(els.upscaleRange.value) });
  });
  els.applyBtn.addEventListener("click", async () => {
    if (activeTab?.id) await chrome.tabs.reload(activeTab.id);
    window.close();
  });
  els.clearSeriesBtn.addEventListener("click", async () => {
    const next = { ...seriesMap };
    delete next[seriesStorageKey];
    await setStorage({ mangashiftSeriesStyles: next });
  });
}

init().catch((error) => {
  updateServerBadge(false, "Error");
  console.error("[MangaShift popup] init failed", error);
});
