// ==================================================================
// Safety Monitor — frontend
// ==================================================================
const EMOTION_LABELS = [
  'Anger', 'Contempt', 'Disgust', 'Fear',
  'Happiness', 'Neutral', 'Sadness', 'Surprise',
];
// CSS-style colours, kept consistent with modules/renderer.py EMOTION_COLOURS.
const EMOTION_COLOURS = {
  Happiness: '#e6dc3c',
  Surprise:  '#f0b43c',
  Neutral:   '#c8c8c8',
  Sadness:   '#3c8cdc',
  Anger:     '#e62828',
  Fear:      '#c83cb4',
  Disgust:   '#3cb43c',
  Contempt:  '#c87878',
  Unknown:   '#a0a0a0',
};
const EMOTION_GLYPHS = {
  Happiness: '😊',
  Surprise:  '😮',
  Neutral:   '😐',
  Sadness:   '😢',
  Anger:     '😠',
  Fear:      '😨',
  Disgust:   '🤢',
  Contempt:  '😒',
  Unknown:   '❔',
};

const els = {
  videoStream: document.getElementById('videoStream'),
  videoWrap:   document.getElementById('videoWrap'),
  overlay:     document.getElementById('overlay'),
  drawHint:    document.getElementById('drawHint'),
  statusDot:   document.getElementById('statusDot'),
  statusText:  document.getElementById('statusText'),
  btnStart:    document.getElementById('btnStart'),
  btnStop:     document.getElementById('btnStop'),
  btnDraw:     document.getElementById('btnDraw'),
  btnFinish:   document.getElementById('btnFinish'),
  btnCancel:   document.getElementById('btnCancel'),
  btnResetZone:document.getElementById('btnResetZone'),
  scenarioList:document.getElementById('scenarioList'),
  statTotal:   document.getElementById('statTotal'),
  statInside:  document.getElementById('statInside'),
  statAvg:     document.getElementById('statAvg'),
  statLongest: document.getElementById('statLongest'),
  statFps:     document.getElementById('statFps'),
  statLat:     document.getElementById('statLat'),
  zoneStats:   document.getElementById('zoneStats'),
  emotionPanel:document.getElementById('emotionPanel'),
  statFaces:   document.getElementById('statFaces'),
  statEmotion: document.getElementById('statEmotion'),
  statFps2:    document.getElementById('statFps2'),
  statLat2:    document.getElementById('statLat2'),
  emotionChart:document.getElementById('emotionChart'),
  chkAnon:     document.getElementById('chkAnon'),
  chkDebug:    document.getElementById('chkDebug'),
};

let state = {
  running: false,
  scenario: 1,
  scenarios: {1: 'Scenario 1', 2: 'Scenario 2', 3: 'Scenario 3'},
  zone: [],            // current committed polygon (in frame coords)
  drawingPoly: null,   // array of [x, y] in frame coords while drawing
  frameSize: [1280, 720],
};

// ------------------------------------------------------------------
// API helpers
// ------------------------------------------------------------------
async function api(path, body) {
  const opts = body !== undefined
    ? { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) }
    : { method: 'GET' };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    applyStatus(s);
  } catch (e) {
    console.error(e);
  }
}

function applyStatus(s) {
  state.running = s.running;
  state.scenario = s.scenario;
  state.scenarios = s.scenarios;
  state.zone = s.zone || [];
  state.frameSize = s.frame_size || state.frameSize;

  els.statusDot.classList.toggle('running', s.running);
  els.statusText.textContent = s.running ? 'running' : 'stopped';

  els.btnStart.disabled = s.running;
  els.btnStop.disabled  = !s.running;

  // Stats — toggle which panel is shown based on active scenario.
  const st = s.stats || {};
  const lat = s.latencies || {};
  const latStr = Object.entries(lat).map(([k,v]) => `${k}:${v}`).join('  ') || '–';

  if (s.scenario === 2) {
    els.zoneStats.classList.add('hidden');
    els.emotionPanel.classList.remove('hidden');

    els.statFaces.textContent   = st.n_faces ?? '–';
    const dom = st.dominant_emotion || '–';
    const glyph = EMOTION_GLYPHS[dom] || '';
    els.statEmotion.textContent = `${dom} ${glyph}`.trim();
    els.statFps2.textContent    = s.fps ?? '–';
    els.statLat2.textContent    = latStr;

    drawEmotionChart(st.emotion_history || []);
  } else {
    els.emotionPanel.classList.add('hidden');
    els.zoneStats.classList.remove('hidden');

    els.statTotal.textContent   = st.total_count ?? '–';
    els.statInside.textContent  = st.currently_inside ?? '–';
    els.statAvg.textContent     = (st.average_dwell_seconds ?? '–') + 's';
    els.statLongest.textContent = (st.longest_dwell_seconds ?? '–') + 's';
    els.statFps.textContent     = s.fps ?? '–';
    els.statLat.textContent     = latStr;
  }

  els.chkAnon.checked  = !!s.anonymise;
  els.chkDebug.checked = !!s.debug;

  renderScenarios();
  if (state.drawingPoly === null) drawOverlay();
}

function renderScenarios() {
  els.scenarioList.innerHTML = '';
  Object.entries(state.scenarios).forEach(([id, name]) => {
    const b = document.createElement('button');
    b.textContent = `${id}. ${name}`;
    if (Number(id) === state.scenario) b.classList.add('active');
    b.onclick = async () => {
      const s = await api('/api/scenario', { scenario: Number(id) });
      applyStatus(s);
    };
    els.scenarioList.appendChild(b);
  });
}

// ------------------------------------------------------------------
// Video stream control
// ------------------------------------------------------------------
function startStream() {
  // Cache-buster prevents the browser from showing a stale frame
  els.videoStream.src = `/video_feed?ts=${Date.now()}`;
}
function stopStream() {
  els.videoStream.src = '';
}

// ------------------------------------------------------------------
// Overlay drawing (polygon)
// ------------------------------------------------------------------
function fitOverlay() {
  // Match overlay canvas to displayed video size.
  const w = els.videoStream.clientWidth || els.videoWrap.clientWidth;
  const h = els.videoStream.clientHeight || els.videoWrap.clientHeight;
  els.overlay.width  = w;
  els.overlay.height = h;
  els.overlay.style.width  = w + 'px';
  els.overlay.style.height = h + 'px';
  // Centre overlay over the <img>
  const r = els.videoStream.getBoundingClientRect();
  const wr = els.videoWrap.getBoundingClientRect();
  els.overlay.style.left = (r.left - wr.left) + 'px';
  els.overlay.style.top  = (r.top  - wr.top)  + 'px';
}

function frameToScreen(pt) {
  const [fw, fh] = state.frameSize;
  const sw = els.overlay.width, sh = els.overlay.height;
  return [pt[0] * sw / fw, pt[1] * sh / fh];
}
function screenToFrame(x, y) {
  const [fw, fh] = state.frameSize;
  const sw = els.overlay.width, sh = els.overlay.height;
  return [Math.round(x * fw / sw), Math.round(y * fh / sh)];
}

function drawOverlay() {
  fitOverlay();
  const ctx = els.overlay.getContext('2d');
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);

  // Show in-progress polygon while drawing
  const poly = state.drawingPoly;
  if (poly && poly.length > 0) {
    ctx.strokeStyle = '#ffd166';
    ctx.fillStyle   = 'rgba(255, 209, 102, 0.15)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    poly.forEach((p, i) => {
      const [x, y] = frameToScreen(p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (poly.length >= 3) { ctx.fill(); }
    poly.forEach(p => {
      const [x, y] = frameToScreen(p);
      ctx.fillStyle = '#ffd166';
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    });
  }
}

function startDraw() {
  state.drawingPoly = [];
  els.overlay.classList.add('drawing');
  els.drawHint.classList.add('visible');
  els.btnDraw.disabled = true;
  els.btnFinish.disabled = false;
  els.btnCancel.disabled = false;
  drawOverlay();
}

function cancelDraw() {
  state.drawingPoly = null;
  els.overlay.classList.remove('drawing');
  els.drawHint.classList.remove('visible');
  els.btnDraw.disabled = false;
  els.btnFinish.disabled = true;
  els.btnCancel.disabled = true;
  drawOverlay();
}

async function finishDraw() {
  const poly = state.drawingPoly;
  if (!poly || poly.length < 3) {
    alert('Need at least 3 points');
    return;
  }
  try {
    const s = await api('/api/zone', { polygon: poly });
    cancelDraw();
    applyStatus(s);
  } catch (e) {
    alert('Failed to set zone: ' + e.message);
  }
}

els.overlay.addEventListener('click', (ev) => {
  if (state.drawingPoly === null) return;
  const r = els.overlay.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  state.drawingPoly.push(screenToFrame(x, y));
  drawOverlay();
});
els.overlay.addEventListener('dblclick', () => {
  if (state.drawingPoly !== null) finishDraw();
});
window.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && state.drawingPoly !== null) cancelDraw();
});
window.addEventListener('resize', drawOverlay);
els.videoStream.addEventListener('load', drawOverlay);

// ------------------------------------------------------------------
// Buttons
// ------------------------------------------------------------------
els.btnStart.onclick = async () => {
  els.btnStart.disabled = true;
  startStream();
  const s = await api('/api/start', {});
  applyStatus(s);
};
els.btnStop.onclick = async () => {
  const s = await api('/api/stop', {});
  applyStatus(s);
  stopStream();
};
els.btnDraw.onclick   = startDraw;
els.btnFinish.onclick = finishDraw;
els.btnCancel.onclick = cancelDraw;
els.btnResetZone.onclick = async () => {
  // Default polygon — same as config.ROI_POLYGON
  const [fw, fh] = state.frameSize;
  const poly = [
    [Math.round(fw*0.25), Math.round(fh*0.25)],
    [Math.round(fw*0.75), Math.round(fh*0.25)],
    [Math.round(fw*0.75), Math.round(fh*0.85)],
    [Math.round(fw*0.25), Math.round(fh*0.85)],
  ];
  const s = await api('/api/zone', { polygon: poly });
  applyStatus(s);
};
els.chkAnon.onchange  = async (e) => {
  await api('/api/anonymise', { value: e.target.checked });
};
els.chkDebug.onchange = async (e) => {
  await api('/api/debug', { value: e.target.checked });
};

// ------------------------------------------------------------------
// Emotion chart (Scenario 2)
// ------------------------------------------------------------------
function drawEmotionChart(history) {
  const cv = els.emotionChart;
  if (!cv) return;
  // Resize backing store to displayed size for crisp text.
  const cssW = cv.clientWidth || cv.width;
  const cssH = cv.clientHeight || cv.height;
  if (cv.width !== cssW || cv.height !== cssH) {
    cv.width = cssW;
    cv.height = cssH;
  }
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);

  const counts = {};
  EMOTION_LABELS.forEach(l => counts[l] = 0);
  let total = 0;
  for (const lbl of history) {
    if (lbl && lbl !== 'Unknown' && counts[lbl] !== undefined) {
      counts[lbl] += 1;
      total += 1;
    }
  }

  const labels = EMOTION_LABELS;
  const padL = 70, padR = 8, padT = 4, padB = 4;
  const rowH = (H - padT - padB) / labels.length;
  const barMaxW = W - padL - padR;
  const denom = Math.max(1, total);

  ctx.font = '11px -apple-system, sans-serif';
  ctx.textBaseline = 'middle';

  labels.forEach((lbl, i) => {
    const y = padT + i * rowH;
    const cy = y + rowH / 2;
    const frac = counts[lbl] / denom;
    const bw = Math.round(barMaxW * frac);

    ctx.fillStyle = '#8b949e';
    ctx.textAlign = 'right';
    ctx.fillText(lbl, padL - 6, cy);

    // background track
    ctx.fillStyle = '#21262d';
    ctx.fillRect(padL, y + 2, barMaxW, rowH - 4);

    // bar
    ctx.fillStyle = EMOTION_COLOURS[lbl] || '#888';
    ctx.fillRect(padL, y + 2, bw, rowH - 4);

    // pct text
    if (total > 0) {
      ctx.fillStyle = '#e6edf3';
      ctx.textAlign = 'left';
      const pct = (100 * frac).toFixed(0) + '%';
      ctx.fillText(pct, padL + bw + 4, cy);
    }
  });
}

// ------------------------------------------------------------------
// Polling loop for status / stats
// ------------------------------------------------------------------
refreshStatus();
setInterval(refreshStatus, 1000);

// If pipeline already running on first load, start streaming immediately.
api('/api/status').then(s => { if (s.running) startStream(); });

