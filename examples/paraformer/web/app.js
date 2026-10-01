const TARGET_RATE = 16000;

const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const recordButton = document.getElementById("record");
const timer = document.getElementById("timer");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const copyButton = document.getElementById("copy");
const healthEl = document.getElementById("health");
const speedEl = document.getElementById("speed");

const api = (path) => new URL(path, document.baseURI).href;

let busy = false;
let recording = false;
let sessionId = null;
let eventSource = null;
let segmentText = "";
let sawSegments = false;
let audioContext = null;
let sourceNode = null;
let processor = null;
let mediaStream = null;
let audioQueue = [];
let pending = [];
let pendingSamples = 0;
let sending = false;
let startedAt = 0;
let timerId = 0;
let finished = false;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setBusy(value) {
  busy = value;
  recordButton.disabled = value;
  fileInput.disabled = value;
}

function showSpeed(data) {
  if (!data || !data.audio_ms) {
    speedEl.textContent = "";
    return;
  }
  speedEl.textContent = `RTF ${Number(data.rtf).toFixed(3)} · 音频 ${data.audio_ms} ms · 耗时 ${data.elapsed_ms} ms`;
}

function floatToInt16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return out;
}

function resample(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const frac = position - index;
    const a = input[index] || 0;
    const b = input[index + 1] !== undefined ? input[index + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function tick() {
  const seconds = Math.floor((Date.now() - startedAt) / 1000);
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  timer.textContent = `${mm}:${ss}`;
}

const SEND_SAMPLES = 4096;

function onAudio(event) {
  const input = event.inputBuffer.getChannelData(0);
  const samples = resample(input, audioContext.sampleRate, TARGET_RATE);
  pending.push(floatToInt16(samples));
  pendingSamples += samples.length;
  if (pendingSamples >= SEND_SAMPLES) flushPending();
}

function flushPending() {
  if (!pending.length) return;
  const total = pending.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Int16Array(total);
  let offset = 0;
  for (const chunk of pending) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  pending = [];
  pendingSamples = 0;
  if (merged.length) {
    audioQueue.push(merged.buffer);
    pumpAudio();
  }
}

async function pumpAudio() {
  if (sending) return;
  sending = true;
  try {
    while (audioQueue.length && sessionId) {
      const buffer = audioQueue.shift();
      try {
        await fetch(api(`api/session/${sessionId}/audio`), {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: buffer,
        });
      } catch (error) {
        setStatus("音频发送失败：" + (error.message || error), true);
      }
    }
  } finally {
    sending = false;
  }
}

function openEvents() {
  eventSource = new EventSource(api(`api/session/${sessionId}/events`));
  eventSource.addEventListener("partial", (event) => {
    const data = JSON.parse(event.data);
    if (sawSegments) {
      return;
    }
    resultEl.value = data.text || "";
    setStatus("识别中…");
  });
  eventSource.addEventListener("segment", (event) => {
    const data = JSON.parse(event.data);
    sawSegments = true;
    const who = data.speaker >= 0 ? `说话人${data.speaker} ` : "";
    const line = `[${(data.start_ms / 1000).toFixed(2)}-${(data.end_ms / 1000).toFixed(2)}s] ${who}${data.text}`;
    segmentText += (segmentText ? "\n" : "") + line;
    resultEl.value = segmentText;
    setStatus("识别中…");
  });
  eventSource.addEventListener("diarization", (event) => {
    const data = JSON.parse(event.data);
    const segments = data.segments || [];
    segmentText = segments
      .map((seg) => {
        const who = seg.speaker >= 0 ? `说话人${seg.speaker} ` : "";
        return `[${(seg.start_ms / 1000).toFixed(2)}-${(seg.end_ms / 1000).toFixed(2)}s] ${who}${seg.text}`;
      })
      .join("\n");
    sawSegments = true;
    resultEl.value = segmentText;
  });
  eventSource.addEventListener("done", (event) => {
    const data = JSON.parse(event.data);
    resultEl.value = sawSegments ? segmentText : (data.text || "");
    copyButton.disabled = !resultEl.value;
    showSpeed(data);
    setStatus(data.status === 0 ? "完成" : `识别失败（status ${data.status}）`, data.status !== 0);
    finished = true;
    cleanup();
  });
  eventSource.onerror = (event) => {
    if (typeof event.data === "string" && event.data) {
      const data = JSON.parse(event.data);
      setStatus(data.error || `runner 错误 ${data.status}`, true);
      finished = true;
      cleanup();
    } else if (sessionId && !finished) {
      setStatus("连接中断", true);
      cleanup();
    }
  };
}

async function releaseMic() {
  try {
    if (processor) {
      processor.onaudioprocess = null;
      processor.disconnect();
      processor = null;
    }
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (audioContext) {
      await audioContext.close();
      audioContext = null;
    }
  } catch (error) {
    setStatus("释放麦克风出错：" + (error.message || error), true);
  }
}

async function startRecording() {
  setBusy(true);
  setStatus("请求麦克风…");
  resultEl.value = "";
  segmentText = "";
  sawSegments = false;
  speedEl.textContent = "";
  copyButton.disabled = true;
  finished = false;
  audioQueue = [];
  pending = [];
  pendingSamples = 0;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (error) {
    setStatus("无法访问麦克风：" + (error.message || error), true);
    setBusy(false);
    return;
  }
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  try {
    audioContext = new AudioCtx({ sampleRate: TARGET_RATE });
  } catch (error) {
    audioContext = new AudioCtx();
  }
  await audioContext.resume();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(2048, 1, 1);
  processor.onaudioprocess = onAudio;
  const mute = audioContext.createGain();
  mute.gain.value = 0;
  setStatus("连接中…");
  try {
    const response = await fetch(api("api/session"), { method: "POST" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    sessionId = data.id;
  } catch (error) {
    setStatus("无法创建会话：" + (error.message || error), true);
    await releaseMic();
    setBusy(false);
    return;
  }
  openEvents();
  sourceNode.connect(processor);
  processor.connect(mute);
  mute.connect(audioContext.destination);
  startedAt = Date.now();
  timer.textContent = "00:00";
  timerId = setInterval(tick, 250);
  recording = true;
  setBusy(false);
  recordButton.classList.add("recording");
  recordButton.textContent = "停止录音";
  setStatus("录音中…");
}

async function stopRecording() {
  recording = false;
  clearInterval(timerId);
  recordButton.classList.remove("recording");
  recordButton.textContent = "开始录音";
  await releaseMic();
  flushPending();
  while (sending || audioQueue.length) {
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
  setStatus("识别中…");
  try {
    await fetch(api(`api/session/${sessionId}/end`), { method: "POST" });
  } catch (error) {
    setStatus("结束失败：" + (error.message || error), true);
  }
}

function cleanup() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (sessionId) {
    const id = sessionId;
    sessionId = null;
    fetch(api(`api/session/${id}`), { method: "DELETE" }).catch(() => {});
  }
  recording = false;
  clearInterval(timerId);
  recordButton.classList.remove("recording");
  recordButton.textContent = "开始录音";
  setBusy(false);
}

recordButton.addEventListener("click", () => {
  if (busy) return;
  if (recording) {
    stopRecording();
  } else {
    startRecording();
  }
});

async function decodeToPcm16(arrayBuffer) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const decodeCtx = new AudioCtx();
  let decoded;
  try {
    decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    decodeCtx.close();
  }
  const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_RATE));
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const offline = new OfflineCtx(1, frames, TARGET_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return floatToInt16(rendered.getChannelData(0));
}

async function transcribeFile(file) {
  if (!file || busy) return;
  setBusy(true);
  setStatus(`读取 ${file.name} …`);
  resultEl.value = "";
  segmentText = "";
  sawSegments = false;
  speedEl.textContent = "";
  copyButton.disabled = true;
  try {
    const pcm = await decodeToPcm16(await file.arrayBuffer());
    setStatus("识别中…");
    const response = await fetch(api("api/transcribe"), {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: pcm.buffer,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (data.status !== 0) throw new Error(`识别失败（runner status ${data.status}）`);
    resultEl.value = data.text || "";
    copyButton.disabled = !resultEl.value;
    showSpeed(data);
    setStatus(data.text ? "完成" : "未识别到文字");
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    setBusy(false);
  }
}

fileInput.addEventListener("change", () => {
  transcribeFile(fileInput.files[0]);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((type) => {
  drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
});
["dragleave", "drop"].forEach((type) => {
  drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.classList.remove("over");
  });
});
drop.addEventListener("drop", (event) => {
  transcribeFile(event.dataTransfer.files[0]);
});

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(resultEl.value);
    setStatus("已复制");
  } catch (error) {
    setStatus("复制失败：" + (error.message || error), true);
  }
});

async function checkHealth() {
  try {
    const response = await fetch(api("api/health"), { cache: "no-store" });
    const data = await response.json();
    healthEl.textContent = data.runner ? "runner 已连接" : "runner 未连接";
    healthEl.className = "health " + (data.runner ? "ok" : "bad");
  } catch (error) {
    healthEl.textContent = "后端不可用";
    healthEl.className = "health bad";
  }
}

checkHealth();
