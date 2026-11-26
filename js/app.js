// Deconvolution Studio - browser UI.  Heavy work runs in Web Workers (worker.js).
import { gaussianPsf, diskPsf, motionPsf } from './lib/psf.js';
import { normalizePsf, luminance } from './lib/core.js';

const $ = (id) => document.getElementById(id);
const S = {
  name: '', srcBitmap: null,         // the loaded image (full resolution)
  original: null,                    // channels at working size
  reference: null,                   // sharp image while a test blur is active
  observed: null, result: null,      // channels
  psf: null, psfDesc: '',
  busy: false,
};

// ================================================================ workers
const POOL = Math.max(1, Math.min(3, navigator.hardwareConcurrency || 2));
const workers = [];
let jobId = 0;
const pending = new Map();
function getWorker(i) {
  if (!workers[i]) {
    const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (ev) => {
      const job = pending.get(ev.data.id);
      if (!job) return;
      if (ev.data.progress !== undefined) { job.onProgress?.(ev.data.progress, ev.data.msg); return; }
      pending.delete(ev.data.id);
      ev.data.error ? job.reject(new Error(ev.data.error)) : job.resolve(ev.data.result);
    };
    w.onerror = (e) => { for (const [id, j] of pending) if (j.w === w) { pending.delete(id); j.reject(new Error(e.message || 'Worker failed to start')); } };
    workers[i] = w;
  }
  return workers[i];
}
function call(i, msg, onProgress, transfer = []) {
  const id = ++jobId, w = getWorker(i % POOL);
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress, w });
    w.postMessage({ id, ...msg }, transfer);
  });
}
const copyCh = (c) => ({ h: c.h, w: c.w, d: Float64Array.from(c.d) });

// ================================================================ status
function setStatus(msg, kind = '') {
  $('status').textContent = msg;
  document.querySelector('.status').classList.toggle('error', kind === 'error');
}
function setProgress(f) {
  if (f == null) { $('bar').hidden = true; return; }
  $('bar').hidden = false;
  $('bar-fill').style.width = `${Math.round(Math.min(1, Math.max(0, f)) * 100)}%`;
}
function setBusy(b) {
  S.busy = b;
  for (const id of ['btn-run', 'btn-psf', 'btn-sim', 'btn-restore', 'btn-reuse']) $(id).disabled = b || (id === 'btn-reuse' && !S.result);
  if (!b) setProgress(null);
}
async function guarded(fn) {
  if (S.busy) return;
  setBusy(true);
  try { await fn(); } catch (e) { console.error(e); setStatus(e.message, 'error'); } finally { setBusy(false); }
}

// ================================================================ images
function channelsFromImageData(id) {
  const { width: w, height: h, data } = id, N = w * h;
  const r = new Float64Array(N), g = new Float64Array(N), b = new Float64Array(N);
  let gray = true;
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    r[i] = data[j] / 255; g[i] = data[j + 1] / 255; b[i] = data[j + 2] / 255;
    if (gray && (data[j] !== data[j + 1] || data[j + 1] !== data[j + 2])) gray = false;
  }
  return gray ? [{ h, w, d: r }] : [{ h, w, d: r }, { h, w, d: g }, { h, w, d: b }];
}
function toCanvas(chs) {
  const { h, w } = chs[0], c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d'), id = ctx.createImageData(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    for (let k = 0; k < 3; k++) {
      const v = chs[chs.length === 1 ? 0 : k].d[i];
      id.data[j + k] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    id.data[j + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return c;
}
function sampleBitmap(bmp, maxDim) {
  let w = bmp.width, h = bmp.height;
  if (maxDim && Math.max(w, h) > maxDim) { const s = maxDim / Math.max(w, h); w = Math.round(w * s); h = Math.round(h * s); }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, w, h);
  return channelsFromImageData(ctx.getImageData(0, 0, w, h));
}

async function loadBlob(blob, name) {
  let bmp;
  try { bmp = await createImageBitmap(blob); }
  catch { throw new Error(`Could not read “${name}” as an image. Use PNG, JPEG, WebP, BMP or GIF.`); }
  S.srcBitmap = bmp; S.name = name;
  applyWorkingSize();
  setStatus(`Opened ${name}. Apply a test blur, or get the PSF and deblur.`);
}
function applyWorkingSize() {
  if (!S.srcBitmap) return;
  const maxDim = +$('work-size').value || 0;
  S.original = sampleBitmap(S.srcBitmap, maxDim);
  S.reference = null; S.observed = S.original; S.result = null;
  const { w, h } = S.original[0], full = `${S.srcBitmap.width}×${S.srcBitmap.height}`;
  $('file-info').textContent = `${S.name}: ${full}, ${S.original.length === 1 ? 'grayscale' : 'colour'}` +
    (w !== S.srcBitmap.width ? `, working at ${w}×${h}` : '');
  setPsf(null, '');
  showMetrics(null);
  viewer.set(S.observed, null, true);
  setMode('input');
}

// ================================================================ viewer
const viewer = (() => {
  const cv = $('view'), vp = $('viewport'), ctx = cv.getContext('2d');
  let left = null, right = null, mode = 'input', zoom = 1, cx = 0, cy = 0, split = 0.5;
  const pointers = new Map();
  let drag = null;

  function size() {
    const r = vp.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    return { W: r.width, H: r.height, dpr };
  }
  function base() { return left || right; }
  function paneW(W) { return mode === 'side' ? (W - 4) / 2 : W; }

  function drawImg(im, x0, W, H, dpr) {
    ctx.save();
    ctx.beginPath(); ctx.rect(x0 * dpr, 0, W * dpr, H * dpr); ctx.clip();
    ctx.imageSmoothingEnabled = zoom < 2;
    const s = zoom * dpr;
    ctx.setTransform(s, 0, 0, s, (x0 + W / 2) * dpr - cx * s, (H / 2) * dpr - cy * s);
    ctx.drawImage(im, 0, 0);
    ctx.restore();
  }

  function render() {
    const { W, H, dpr } = size();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    $('empty').hidden = !!base();
    $('tag-l').textContent = ''; $('tag-r').textContent = '';
    if (!base()) return;
    const L = left, Rr = right;
    if (mode === 'input' || (!Rr && mode !== 'side')) {
      drawImg(L, 0, W, H, dpr);
      $('tag-l').textContent = mode === 'input' ? 'Input' : 'Input (no result yet)';
    } else if (mode === 'result') {
      drawImg(Rr, 0, W, H, dpr); $('tag-l').textContent = 'Result';
    } else if (mode === 'split') {
      const sx = split * W;
      ctx.save(); ctx.beginPath(); ctx.rect(sx * dpr, 0, (W - sx) * dpr, H * dpr); ctx.clip();
      drawImg(Rr, 0, W, H, dpr); ctx.restore();
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, sx * dpr, H * dpr); ctx.clip();
      drawImg(L, 0, W, H, dpr); ctx.restore();
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--laser').trim() || '#d7263d';
      ctx.fillRect((sx - 1) * dpr, 0, 2 * dpr, H * dpr);
      ctx.beginPath(); ctx.arc(sx * dpr, (H / 2) * dpr, 11 * dpr, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = `${11 * dpr}px Archivo, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('◂ ▸', sx * dpr, (H / 2) * dpr);
      $('tag-l').textContent = 'Input'; $('tag-r').textContent = 'Result';
    } else {
      const pw = paneW(W);
      drawImg(L, 0, pw, H, dpr);
      if (Rr) drawImg(Rr, pw + 4, pw, H, dpr);
      $('tag-l').textContent = 'Input'; $('tag-r').textContent = Rr ? 'Result' : 'Result (none yet)';
    }
    $('zoom-o').textContent = `${Math.round(zoom * 100)}%`;
  }

  function fit() {
    const im = base(); if (!im) { render(); return; }
    const r = vp.getBoundingClientRect(), W = paneW(r.width);
    zoom = Math.min(W / im.width, r.height / im.height, 8);
    cx = im.width / 2; cy = im.height / 2;
    render();
  }
  function zoomAt(x, y, f) {
    if (!base()) return;
    const r = vp.getBoundingClientRect(); let W = paneW(r.width);
    if (mode === 'side' && x > W) x -= W + 4;
    const ix = cx + (x - W / 2) / zoom, iy = cy + (y - r.height / 2) / zoom;
    zoom = Math.min(32, Math.max(0.05, zoom * f));
    cx = ix - (x - W / 2) / zoom; cy = iy - (y - r.height / 2) / zoom;
    render();
  }
  const nearSplit = (x) => mode === 'split' && right && Math.abs(x - split * vp.getBoundingClientRect().width) < 14;

  cv.addEventListener('wheel', (e) => { e.preventDefault(); const r = cv.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.2 : 1 / 1.2); }, { passive: false });
  cv.addEventListener('dblclick', fit);
  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId);
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    pointers.set(e.pointerId, { x, y });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      drag = { pinch: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
    } else if (nearSplit(x)) drag = { split: true };
    else { drag = { x, y, cx, cy }; cv.classList.add('dragging'); }
  });
  cv.addEventListener('pointermove', (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x, y });
    cv.classList.toggle('splitcur', !drag && nearSplit(x));
    if (!drag) return;
    if (drag.pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
      zoomAt(drag.mx, drag.my, d / drag.pinch); drag.pinch = d;
    } else if (drag.split) { split = Math.min(1, Math.max(0, x / r.width)); render(); }
    else if (drag.x !== undefined) { cx = drag.cx - (x - drag.x) / zoom; cy = drag.cy - (y - drag.y) / zoom; render(); }
  });
  const end = (e) => { pointers.delete(e.pointerId); if (pointers.size === 0) { drag = null; cv.classList.remove('dragging'); } };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);
  new ResizeObserver(() => render()).observe(vp);

  return {
    set(inp, res, refit) {
      const before = left ? [left.width, left.height] : null;
      left = inp ? toCanvas(inp) : null;
      right = res ? toCanvas(res) : null;
      if (refit || !before || before[0] !== left?.width || before[1] !== left?.height) fit(); else render();
    },
    mode(m) { mode = m; render(); },
    fit, actual() { zoom = 1; render(); },
  };
})();

function setMode(m) {
  document.querySelectorAll('.seg button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === m)));
  viewer.mode(m);
}

// ================================================================ PSF panel
const INFERNO = [[0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 165, 10], [246, 215, 70], [252, 255, 164]];
function inferno(v) {
  const t = Math.min(1, Math.max(0, v)) * (INFERNO.length - 1), i = Math.min(INFERNO.length - 2, Math.floor(t)), f = t - i;
  return INFERNO[i].map((c, k) => Math.round(c + (INFERNO[i + 1][k] - c) * f));
}
function setPsf(psf, desc) {
  S.psf = psf; S.psfDesc = desc;
  const cv = $('psf-canvas'), ctx = cv.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cv.width, cv.height);
  $('psf-desc').textContent = psf ? desc : 'No PSF yet.';
  $('dl-psf-png').disabled = $('dl-psf-csv').disabled = !psf;
  if (!psf) return;
  let mx = 0; for (const v of psf.d) mx = Math.max(mx, v);
  const t = document.createElement('canvas'); t.width = psf.w; t.height = psf.h;
  const tc = t.getContext('2d'), id = tc.createImageData(psf.w, psf.h);
  for (let i = 0; i < psf.d.length; i++) { const [r, g, b] = inferno(psf.d[i] / mx); id.data.set([r, g, b, 255], i * 4); }
  tc.putImageData(id, 0, 0);
  const s = Math.floor(cv.width / Math.max(psf.w, psf.h)) || 1, dw = psf.w * Math.max(1, s), dh = psf.h * Math.max(1, s);
  ctx.imageSmoothingEnabled = false;
  if (s >= 1) ctx.drawImage(t, (cv.width - dw) / 2, (cv.height - dh) / 2, dw, dh);
  else ctx.drawImage(t, 0, 0, cv.width, cv.height);
}

const PSF_HINTS = {
  auto: 'Reads the image spectrum and chooses linear motion, defocus or Gaussian blur, then fits its size.',
  gaussian: 'Soft optical blur. σ in pixels.',
  disk: 'Out-of-focus blur. Radius of the blur disk in pixels.',
  motion: 'Straight motion streak. Angle is counter-clockwise from horizontal.',
  blind: 'Recovers any kernel shape, such as curved camera shake. The kernel size must be larger than the blur. Takes up to about half a minute.',
  point: 'Measures the PSF from stars, beads or small bright dots. The box must contain one whole point image.',
  edge: 'Measures a Gaussian PSF from the sharpest straight high-contrast edge in the image.',
  file: 'Load a measured PSF: a grayscale image, or a CSV / text matrix.',
};
const PARAMS = {
  gaussian: [['sigma', 'σ (px)', 2.0, 0.3, 30, 0.1]],
  disk: [['radius', 'Radius (px)', 5, 0.5, 60, 0.5]],
  motion: [['length', 'Length (px)', 15, 1, 150, 0.5], ['angle', 'Angle (°)', 30, 0, 180, 1]],
  blind: [['ksize', 'Kernel size (px)', 31, 9, 101, 2]],
  point: [['box', 'Box size (px)', 21, 7, 101, 2]],
};
function renderPsfParams() {
  const src = $('psf-src').value, box = $('psf-params');
  box.innerHTML = '';
  for (const [key, label, val, min, max, step] of PARAMS[src] || []) {
    const l = document.createElement('label'); l.className = 'field';
    l.innerHTML = `<span>${label}</span><input type="number" id="p-${key}" value="${val}" min="${min}" max="${max}" step="${step}">`;
    box.appendChild(l);
  }
  if (src === 'file') {
    const row = document.createElement('div'); row.className = 'row';
    row.innerHTML = '<label class="btn" for="psf-file">Choose PSF file…</label><input type="file" id="psf-file" accept="image/*,.csv,.txt" hidden><span class="note" id="psf-file-name"></span>';
    box.appendChild(row);
    $('psf-file').addEventListener('change', (e) => e.target.files[0] && loadPsfFile(e.target.files[0]));
  }
  $('psf-hint').textContent = PSF_HINTS[src];
}
const pval = (k) => +$('p-' + k).value;

async function loadPsfFile(file) {
  try {
    let psf;
    if (/\.(csv|txt)$/i.test(file.name)) {
      const rows = (await file.text()).trim().split(/\r?\n/).map((l) => l.trim().split(/[\s,;]+/).map(Number));
      const w = rows[0].length;
      if (rows.some((r) => r.length !== w || r.some(isNaN))) throw new Error('The PSF file must be a rectangular matrix of numbers.');
      psf = { h: rows.length, w, d: Float64Array.from(rows.flat()) };
    } else {
      const ch = sampleBitmap(await createImageBitmap(file), 0);
      psf = luminance(ch.map((c) => ({ ...c })));
    }
    psf = normalizePsf(psf);
    setPsf(psf, `From file\n${file.name}\n${psf.w}×${psf.h} px`);
    $('psf-file-name').textContent = file.name;
    setStatus('PSF loaded. Press Deblur.');
  } catch (e) { setStatus(e.message, 'error'); }
}

async function computePsf() {
  const src = $('psf-src').value;
  if (src === 'file') {
    if (!S.psf) throw new Error('Choose a PSF file first.');
    return;
  }
  const params = {};
  for (const [k] of PARAMS[src] || []) params[k] = pval(k);
  const gray = luminance(S.observed);
  setStatus(src === 'blind' ? 'Estimating the blur kernel (coarse to fine)…' : 'Getting the PSF…');
  setProgress(src === 'blind' ? 0 : 0.3);
  const t0 = performance.now();
  const r = await call(0, { op: 'estimate', gray: copyCh(gray), source: src, params }, (f, msg) => { setProgress(f); if (msg) setStatus(msg); });
  setPsf(r.psf, r.desc);
  return (performance.now() - t0) / 1000;
}

// ================================================================ actions
function testPsf() {
  const t = $('tb-type').value, v = +$('tb-size').value;
  if (t === 'gaussian') return [gaussianPsf(v), `Gaussian σ ${v}`];
  if (t === 'disk') return [diskPsf(v), `defocus R ${v}`];
  return [motionPsf(v, +$('tb-angle').value), `motion ${v} px at ${$('tb-angle').value}°`];
}

async function applyTestBlur() {
  if (!S.original) throw new Error('Open an image first.');
  const [psf, desc] = testPsf(), noise = +$('tb-noise').value / 100;
  setStatus(`Applying test blur (${desc})…`); setProgress(0.5);
  const outs = await Promise.all(S.original.map((ch, k) =>
    call(k, { op: 'simulate', ch: copyCh(ch), psf, noise, seed: 1234 + k })));
  S.reference = S.original;
  S.observed = outs.map((o) => o.out);
  S.result = null;
  viewer.set(S.observed, null);
  setMode('input');
  const m = await call(0, { op: 'metrics', ref: S.reference.map(copyCh), x: S.observed.map(copyCh) });
  showMetrics({ blurred: m });
  setStatus(`Test blur applied: ${desc}, noise ${$('tb-noise').value}%. Now get the PSF and deblur.`);
}

async function deblur() {
  if (!S.observed) throw new Error('Open an image first.');
  const t0 = performance.now();
  if (!S.psf) await computePsf();
  const method = $('alg').value, scale = 10 ** +$('reg').value, iterations = +$('iters').value;
  const opts = method === 'rl' || method === 'rl_tv' ? { iterations, scale } : { scale };
  const n = S.observed.length, prog = new Array(n).fill(0);
  setStatus(`Deblurring with ${$('alg').selectedOptions[0].text.replace(' (recommended)', '')}` +
    (n > 1 ? ` (3 colour channels${POOL > 1 ? ' in parallel' : ''})…` : '…'));
  setProgress(0);
  const outs = await Promise.all(S.observed.map((ch, k) =>
    call(k, { op: 'deconvolve', ch: copyCh(ch), psf: { h: S.psf.h, w: S.psf.w, d: Float64Array.from(S.psf.d) }, method, opts },
      (f) => { prog[k] = f; setProgress(prog.reduce((a, b) => a + b, 0) / n); })));
  S.result = outs.map((o) => o.out);
  const dt = (performance.now() - t0) / 1000;
  viewer.set(S.observed, S.result);
  setMode(document.querySelector('.seg [aria-checked="true"]').dataset.mode === 'input' ? 'split' : document.querySelector('.seg [aria-checked="true"]').dataset.mode);
  $('dl-result').disabled = false; $('btn-reuse').disabled = false;
  const extra = { param: outs[0].param, time: dt };
  if (S.reference && S.reference[0].h === S.result[0].h) {
    const [b, r] = await Promise.all([
      call(0, { op: 'metrics', ref: S.reference.map(copyCh), x: S.observed.map(copyCh) }),
      call(1, { op: 'metrics', ref: S.reference.map(copyCh), x: S.result.map(copyCh) })]);
    showMetrics({ blurred: b, result: r, ...extra });
  } else showMetrics(extra);
  setStatus(`Done in ${dt.toFixed(1)} s. Drag the red divider to compare.`);
}

function showMetrics(m) {
  const el = $('metrics');
  if (!m) { el.hidden = true; el.innerHTML = ''; return; }
  let html = '';
  if (m.blurred) {
    html += '<dt></dt><dd>PSNR</dd><dd>SSIM</dd>';
    html += `<dt>Blurred</dt><dd>${m.blurred.psnr.toFixed(2)} dB</dd><dd>${m.blurred.ssim.toFixed(3)}</dd>`;
    if (m.result) html += `<dt>Result</dt><dd class="gain">${m.result.psnr.toFixed(2)} dB</dd><dd class="gain">${m.result.ssim.toFixed(3)}</dd>`;
  }
  if (m.param) html += `<dd class="wide">${m.param}, ${m.time.toFixed(1)} s</dd>`;
  el.innerHTML = html; el.hidden = !html;
}

function download(blob, name) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const baseName = () => (S.name || 'image').replace(/\.[^.]+$/, '');

// ================================================================ wiring
function bindSlider(id, fmt) {
  const el = $(id), out = $(id + '-o');
  const upd = () => { out.textContent = fmt(+el.value); };
  el.addEventListener('input', upd); upd();
}
bindSlider('tb-size', (v) => v.toFixed(1));
bindSlider('tb-angle', (v) => `${v}°`);
bindSlider('tb-noise', (v) => `${v.toFixed(2)}%`);
bindSlider('reg', (v) => `×${(10 ** v).toPrecision(2)}`);
bindSlider('iters', (v) => String(v));

const ALG_HINTS = {
  tv: 'Keeps edges sharp while removing noise. Image borders are treated as unknown, so there is no ringing.',
  rl: 'Standard for astronomy and microscopy. More iterations give a sharper but noisier result.',
  rl_tv: 'Richardson–Lucy with an edge-preserving penalty, so it can run many iterations without noise build-up.',
  tikhonov: 'Instant closed-form filter; the weight is chosen by cross-validation.',
  wiener: 'Fastest closed-form filter; the noise level is estimated from the image.',
};
function algChanged() {
  const a = $('alg').value;
  $('alg-hint').textContent = ALG_HINTS[a] + ' Regularisation: right is smoother, left is sharper.';
  const it = a === 'rl' || a === 'rl_tv';
  $('iters').disabled = !it; $('iter-row').classList.toggle('off', !it);
  if (a === 'rl_tv' && +$('iters').value === 30) { $('iters').value = 50; $('iters').dispatchEvent(new Event('input')); }
}
function tbTypeChanged() {
  const m = $('tb-type').value === 'motion';
  $('tb-angle').disabled = !m; $('tb-angle-row').classList.toggle('off', !m);
}

$('alg').addEventListener('change', algChanged);
$('tb-type').addEventListener('change', tbTypeChanged);
$('psf-src').addEventListener('change', () => { if ($('psf-src').value !== 'file') setPsf(null, ''); renderPsfParams(); });
$('psf-params').addEventListener('input', () => { if (S.psf && $('psf-src').value !== 'file') setPsf(null, ''); });
$('work-size').addEventListener('change', () => guarded(async () => { applyWorkingSize(); setStatus('Working size changed. The PSF and result were cleared.'); }));
$('file-in').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) guarded(() => loadBlob(f, f.name)); e.target.value = ''; });
document.querySelectorAll('[data-sample]').forEach((b) => b.addEventListener('click', () => guarded(async () => {
  setStatus('Loading sample…');
  const r = await fetch(b.dataset.sample);
  await loadBlob(await r.blob(), b.dataset.sample.split('/').pop());
})));
$('btn-sim').addEventListener('click', () => guarded(applyTestBlur));
$('btn-restore').addEventListener('click', () => {
  if (!S.original || S.busy) return;
  S.reference = null; S.observed = S.original; S.result = null;
  viewer.set(S.observed, null); setMode('input'); showMetrics(null);
  $('dl-result').disabled = true; $('btn-reuse').disabled = true;
  setStatus('Original restored.');
});
$('btn-psf').addEventListener('click', () => guarded(async () => {
  if (!S.observed) throw new Error('Open an image first.');
  const dt = await computePsf();
  setStatus(`PSF ready${dt ? ` (${dt.toFixed(1)} s)` : ''}. Press Deblur.`);
}));
$('btn-run').addEventListener('click', () => guarded(deblur));
$('btn-reuse').addEventListener('click', () => {
  if (!S.result || S.busy) return;
  S.observed = S.result; S.result = null;
  viewer.set(S.observed, null); setMode('input');
  $('dl-result').disabled = true; $('btn-reuse').disabled = true;
  setStatus('The result is now the input. Deblur again to refine it.');
});
$('dl-result').addEventListener('click', () => {
  if (!S.result) return;
  toCanvas(S.result).toBlob((b) => download(b, `${baseName()}_deblurred.png`), 'image/png');
});
$('dl-psf-png').addEventListener('click', () => {
  if (!S.psf) return;
  let mx = 0; for (const v of S.psf.d) mx = Math.max(mx, v);
  toCanvas([{ ...S.psf, d: S.psf.d.map((v) => v / mx) }]).toBlob((b) => download(b, 'psf.png'), 'image/png');
});
$('dl-psf-csv').addEventListener('click', () => {
  if (!S.psf) return;
  const rows = [];
  for (let r = 0; r < S.psf.h; r++) rows.push(Array.from(S.psf.d.slice(r * S.psf.w, (r + 1) * S.psf.w), (v) => v.toExponential(6)).join(','));
  download(new Blob([rows.join('\n')], { type: 'text/csv' }), 'psf.csv');
});
document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('btn-fit').addEventListener('click', () => viewer.fit());
$('btn-100').addEventListener('click', () => viewer.actual());
$('btn-help').addEventListener('click', () => $('help').showModal());

document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  const modes = { 1: 'input', 2: 'result', 3: 'split', 4: 'side' };
  if (modes[e.key]) setMode(modes[e.key]);
  else if (e.key === '0') viewer.fit();
});

// drag & drop / paste
const vp = $('viewport');
['dragenter', 'dragover'].forEach((t) => vp.addEventListener(t, (e) => { e.preventDefault(); vp.classList.add('drop'); }));
['dragleave', 'drop'].forEach((t) => vp.addEventListener(t, (e) => { e.preventDefault(); vp.classList.remove('drop'); }));
vp.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) guarded(() => loadBlob(f, f.name)); });
document.addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/'));
  if (f) guarded(() => loadBlob(f, f.name || 'pasted image'));
});

// start
renderPsfParams(); algChanged(); tbTypeChanged(); setPsf(null, '');
guarded(async () => {
  const r = await fetch('images/cameraman.png');
  await loadBlob(await r.blob(), 'cameraman.png');
  setStatus('Ready. Try Apply test blur, then Deblur, or open your own photo.');
});
