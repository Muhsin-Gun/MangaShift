const API_URL = "http://127.0.0.1:8000/process-image";
const MAX_CONCURRENCY = 2;

let active = 0;
const queue = [];

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function enqueue(task) {
  return new Promise((resolve, reject) => {
    queue.push({ task, resolve, reject });
    drain();
  });
}

async function drain() {
  if (active >= MAX_CONCURRENCY || queue.length === 0) return;
  const item = queue.shift();
  active += 1;
  try {
    const result = await item.task();
    item.resolve(result);
  } catch (error) {
    item.reject(error);
  } finally {
    active -= 1;
    drain();
  }
}

async function processImageTask(payload) {
  const {
    bytes,
    imageUrl = "",
    filename = "panel.png",
    sourceLang = "auto",
    targetLang = "en",
    style = "original",
    colorize = false,
    upscale = 1.0,
    preferredStyle = "",
  } = payload;

  let binary = bytes;
  if ((!binary || !binary.length) && imageUrl) {
    const sourceResponse = await fetch(imageUrl, { credentials: "include" });
    if (!sourceResponse.ok) {
      throw new Error(`Failed to fetch source image (${sourceResponse.status})`);
    }
    binary = Array.from(new Uint8Array(await sourceResponse.arrayBuffer()));
  }
  if (!binary || !binary.length) {
    throw new Error("No image bytes provided");
  }

  const fileBlob = new Blob([new Uint8Array(binary)], { type: "image/png" });
  const formData = new FormData();
  formData.append("file", fileBlob, filename);
  formData.append("source_lang", sourceLang);
  formData.append("target_lang", targetLang);
  formData.append("style", style);
  formData.append("colorize", String(Boolean(colorize)));
  formData.append("upscale", String(upscale));

  const headers = {};
  if (preferredStyle) headers["X-Preferred-Style"] = preferredStyle;

  const response = await fetch(API_URL, {
    method: "POST",
    body: formData,
    headers,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Backend failed (${response.status}): ${text}`);
  }
  const outputBuffer = await response.arrayBuffer();
  return {
    imageBase64: arrayBufferToBase64(outputBuffer),
    contentType: "image/png",
    cacheHit: response.headers.get("X-Cache-Hit") === "1",
    report: response.headers.get("X-Process-Report") || "",
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || typeof message !== "object") return false;

  if (message.type === "PROCESS_IMAGE") {
    enqueue(() => processImageTask(message.payload))
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }

  if (message.type === "GET_QUEUE_STATUS") {
    sendResponse({ ok: true, active, pending: queue.length });
    return false;
  }

  return false;
});
