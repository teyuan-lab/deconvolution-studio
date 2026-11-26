// PSF models and estimators - JavaScript port of deconv/psf.py.
import * as C from './core.js';
import { j1, erf, logaddexp, leastSquares } from './mathx.js';

const { img } = C;
const odd = (n) => { n = Math.ceil(n); return n % 2 === 1 ? n : n + 1; };

// ==================================================================== models
export function gaussianPsf(sigma, size = null) {
  sigma = Math.max(+sigma, 1e-3);
  size = size || Math.max(3, odd(6.5 * sigma));
  const r = (size - 1) / 2, p = img(size, size);
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++) {
      const y = i - r, x = j - r;
      p.d[i * size + j] = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
    }
  return C.normalizePsf(p);
}

export function diskPsf(radius, size = null, os = 8) {
  radius = Math.max(+radius, 0.5);
  size = size || odd(2 * radius + 3);
  const r = (size - 1) / 2, p = img(size, size);
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++) {
      let cnt = 0;
      for (let a = 0; a < os; a++)
        for (let b = 0; b < os; b++) {
          const y = i + (a + 0.5) / os - 0.5 - r, x = j + (b + 0.5) / os - 0.5 - r;
          if (x * x + y * y <= radius * radius) cnt++;
        }
      p.d[i * size + j] = cnt / (os * os);
    }
  return C.normalizePsf(p);
}

export function motionPsf(length, angle = 0, size = null, os = 8) {
  length = Math.max(+length, 1);
  size = size || odd(length + 3);
  const r = (size - 1) / 2, t = (angle * Math.PI) / 180, n = size * os;
  let hi = img(n, n);
  const m = Math.ceil(length * os * 4) + 1;
  for (let k = 0; k < m; k++) {
    const s = -length / 2 + (length * k) / (m - 1);
    const xs = (s * Math.cos(t) + r + 0.5) * os - 0.5, ys = (-s * Math.sin(t) + r + 0.5) * os - 0.5;
    const xi = Math.min(n - 1, Math.max(0, Math.round(xs))), yi = Math.min(n - 1, Math.max(0, Math.round(ys)));
    hi.d[yi * n + xi] += 1;
  }
  hi = C.gaussianFilter(hi, os * 0.35);
  const p = img(size, size);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) p.d[Math.floor(i / os) * size + Math.floor(j / os)] += hi.d[i * n + j];
  return C.normalizePsf(p);
}

export function makePsf(kind, prm) {
  if (kind === 'gaussian') return gaussianPsf(prm.sigma);
  if (kind === 'disk') return diskPsf(prm.radius);
  if (kind === 'motion') return motionPsf(prm.length, prm.angle || 0);
  throw new Error('unknown PSF kind ' + kind);
}

// =================================================================== helpers
export function centerPsf(psf) {
  const p = C.normalizePsf(psf);
  const [cy, cx] = C.centerOfMass(p);
  return C.normalizePsf(C.shift(p, (p.h - 1) / 2 - cy, (p.w - 1) / 2 - cx));
}

function window2(h, w) {
  const a = C.hanning(h), b = C.hanning(w), o = img(h, w);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) o.d[r * w + c] = a[r] * b[c];
  return o;
}

function squareCrop(g) {
  let n = Math.min(g.h, g.w); n -= n % 2;
  return C.crop(g, Math.floor((g.h - n) / 2), Math.floor((g.w - n) / 2), n, n);
}

function logSpectrum(g0) {
  const g = squareCrop(g0), n = g.h, m = C.mean(g), win = window2(n, n);
  const x = img(n, n);
  for (let i = 0; i < n * n; i++) x.d[i] = (g.d[i] - m) * win.d[i];
  const X = C.rfft2(x.d, n, n), o = img(n, n);
  for (let i = 0; i < n * n; i++) o.d[i] = Math.log(Math.hypot(X.re[i], X.im[i]) + 1e-8);
  return C.fftshift(o);
}

function radialProfile(a) {
  const n = a.h, c = Math.floor(n / 2), nb = c;
  const sums = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const rb = Math.floor(Math.hypot(y - c, x - c));
      if (rb < nb) { sums[rb] += a.d[y * n + x]; cnt[rb]++; }
    }
  const prof = new Float64Array(nb), f = new Float64Array(nb);
  for (let i = 0; i < nb; i++) { prof[i] = sums[i] / Math.max(cnt[i], 1); f[i] = i / n; }
  return { f, prof };
}

const interp = (x, xp, fp) => { // np.interp for increasing integer xp = 0..len-1
  if (x <= 0) return fp[0];
  if (x >= fp.length - 1) return fp[fp.length - 1];
  const i = Math.floor(x), t = x - i;
  return fp[i] * (1 - t) + fp[i + 1] * t;
};

// ========================================================= 1. point sources
export function findPointSources(g, size = 31, n = 10, saturation = 0.98) {
  const bg = Math.min(g.h, g.w) > 3 * size ? C.medianFilterApprox(g, size) : null;
  const bgv = bg ? null : C.median(g.d);
  const d = img(g.h, g.w);
  for (let i = 0; i < d.d.length; i++) d.d[i] = g.d[i] - (bg ? bg.d[i] : bgv);
  const mx = C.maximumFilter(d, size), sd = C.std(d.d), h = Math.floor(size / 2), cand = [];
  for (let r = h; r < g.h - h; r++)
    for (let c = h; c < g.w - h; c++) {
      const i = r * g.w + c;
      if (d.d[i] === mx.d[i] && d.d[i] > 6 * sd && g.d[i] < saturation) cand.push([d.d[i], r, c]);
    }
  cand.sort((a, b) => b[0] - a[0]);
  const out = [];
  for (const [, r, c] of cand) {
    if (out.every(([r2, c2]) => Math.abs(r - r2) > size || Math.abs(c - c2) > size)) out.push([r, c]);
    if (out.length >= n) break;
  }
  return out;
}

export function estimateFromPointSource(g, size = 31, n = 10) {
  size = odd(size);
  const h = Math.floor(size / 2), centers = findPointSources(g, size, n);
  if (!centers.length) throw new Error('No isolated point source (star / bead / bright dot) was found.');
  const stack = [];
  for (const [r, c] of centers) {
    const crop = C.crop(g, r - h, c - h, size, size);
    const rim = [];
    for (let k = 0; k < size; k++) rim.push(crop.d[k], crop.d[(size - 1) * size + k], crop.d[k * size], crop.d[k * size + size - 1]);
    const bgv = C.median(rim);
    let s = 0;
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const i = y * size + x, rr = Math.hypot(y - h, x - h);
        crop.d[i] = Math.max(crop.d[i] - bgv, 0) * Math.min(Math.max((h + 0.5 - rr) / 2, 0), 1);
        s += crop.d[i];
      }
    if (s > 0) stack.push(centerPsf(crop));
  }
  if (!stack.length) throw new Error('Point-source crops were empty.');
  const med = img(size, size), buf = new Float64Array(stack.length);
  for (let i = 0; i < size * size; i++) {
    for (let k = 0; k < stack.length; k++) buf[k] = stack[k].d[i];
    med.d[i] = C.median(buf);
  }
  return { psf: centerPsf(med), count: stack.length };
}

// =============================================================== 2. edge
function bestEdgeRoi(g, win) {
  const [gy, gx] = C.gradient(C.gaussianFilter(g, 1.0));
  const n = g.d.length, xx = img(g.h, g.w), yy = img(g.h, g.w), xy = img(g.h, g.w);
  for (let i = 0; i < n; i++) { xx.d[i] = gx.d[i] ** 2; yy.d[i] = gy.d[i] ** 2; xy.d[i] = gx.d[i] * gy.d[i]; }
  const jxx = C.uniformFilter(xx, win), jyy = C.uniformFilter(yy, win), jxy = C.uniformFilter(xy, win);
  const h = Math.floor(win / 2);
  let best = -1, br = h, bc = h;
  for (let r = h; r < g.h - h; r++)
    for (let c = h; c < g.w - h; c++) {
      const i = r * g.w + c, tr = jxx.d[i] + jyy.d[i];
      const coh = Math.sqrt((jxx.d[i] - jyy.d[i]) ** 2 + 4 * jxy.d[i] ** 2) / (tr + 1e-12);
      const s = tr * coh ** 4;
      if (s > best) { best = s; br = r; bc = c; }
    }
  return [br - h, bc - h, win, win];
}

export function estimateFromEdge(g) {
  const win = Math.min(64, Math.floor(Math.min(g.h, g.w) / 2));
  const [r0, c0, hh, ww] = bestEdgeRoi(g, win);
  const p = C.crop(g, r0, c0, hh, ww);
  const [gy, gx] = C.gradient(C.gaussianFilter(p, 0.7));
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < p.d.length; i++) { sxx += gx.d[i] ** 2; sxy += gx.d[i] * gy.d[i]; syy += gy.d[i] ** 2; }
  // principal eigenvector of [[sxx, sxy], [sxy, syy]]
  const tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  let nx, ny;
  if (Math.abs(sxy) > 1e-15) { nx = l1 - syy; ny = sxy; } else if (sxx >= syy) { nx = 1; ny = 0; } else { nx = 0; ny = 1; }
  const nn = Math.hypot(nx, ny); nx /= nn; ny /= nn;
  let ms = 0, my = 0, mxx = 0;
  for (let r = 0; r < hh; r++)
    for (let c = 0; c < ww; c++) {
      const i = r * ww + c, m = Math.hypot(gx.d[i], gy.d[i]);
      ms += m; my += m * r; mxx += m * c;
    }
  const cy = my / ms, cx = mxx / ms, binw = 0.25, lim = Math.min(hh, ww) / 2 - 1;
  const bins = [], vals = [];
  for (let r = 0; r < hh; r++)
    for (let c = 0; c < ww; c++) {
      const d = (c - cx) * nx + (r - cy) * ny;
      if (Math.abs(d) < lim) { bins.push(Math.round(d / binw)); vals.push(p.d[r * ww + c]); }
    }
  const off = Math.min(...bins), nb = Math.max(...bins) - off + 1;
  const s = new Float64Array(nb), cnt = new Float64Array(nb);
  bins.forEach((b, i) => { s[b - off] += vals[i]; cnt[b - off]++; });
  const dist = Array.from({ length: nb }, (_, i) => (i + off) * binw);
  const gi = [...Array(nb).keys()].filter((i) => cnt[i] > 0);
  const esf = dist.map((x, i) => {
    if (cnt[i] > 0) return s[i] / cnt[i];
    let a = gi[0], b = gi[gi.length - 1];
    for (const k of gi) { if (k < i) a = k; if (k > i) { b = k; break; } }
    if (i < a) return s[a] / cnt[a];
    if (i > b) return s[b] / cnt[b];
    const t = (i - a) / (b - a);
    return (s[a] / cnt[a]) * (1 - t) + (s[b] / cnt[b]) * t;
  });
  const model = (q, x) => q[0] + (q[1] - q[0]) * 0.5 * (1 + erf((x - q[2]) / (Math.SQRT2 * q[3])));
  const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const q0 = [avg(esf.slice(0, 5)), avg(esf.slice(-5)), 0, 1.5];
  const res = leastSquares((q) => dist.map((x, i) => model(q, x) - esf[i]), q0,
    [-Infinity, -Infinity, dist[0], 0.2], [Infinity, Infinity, dist[nb - 1], 50]);
  const sigma = Math.sqrt(Math.max(res.x[3] ** 2 - 0.25 ** 2 / 12, 0.04));
  return { psf: gaussianPsf(sigma), sigma, roi: [r0, c0, hh, ww] };
}

// ============================================== 3. blind parametric (spectrum)
export function estimateGaussianBlind(g) {
  const ls = logSpectrum(g), ls2 = img(ls.h, ls.w);
  for (let i = 0; i < ls.d.length; i++) ls2.d[i] = 2 * ls.d[i];
  const { f: f0, prof } = radialProfile(ls2);
  const f = [], lp = [];
  for (let i = 0; i < f0.length; i++) if (f0[i] > 0.01 && f0[i] < 0.5) { f.push(f0[i]); lp.push(prof[i]); }
  const model = (q) => f.map((fi) => logaddexp(q[0] - q[1] * Math.log(fi) - 4 * Math.PI ** 2 * q[2] ** 2 * fi * fi, q[3]));
  let best = null;
  for (const s0 of [0.7, 1.5, 3.0, 6.0]) {
    const q0 = [lp[0] + 2 * Math.log(f[0]), 2.0, s0, lp[lp.length - 1]];
    const r = leastSquares((q) => model(q).map((v, i) => v - lp[i]), q0, [-Infinity, 1.0, 0.0, -Infinity], [Infinity, 3.5, 30, Infinity]);
    if (!best || r.cost < best.cost) best = r;
  }
  const sigma = best.x[2];
  return { psf: gaussianPsf(Math.max(sigma, 0.3)), sigma };
}

function whitenedRadial(g) {
  const { f: f0, prof: p0 } = radialProfile(logSpectrum(g));
  const f = [], prof = [];
  for (let i = 0; i < f0.length; i++) if (f0[i] > 0.005 && f0[i] < 0.45) { f.push(f0[i]); prof.push(p0[i]); }
  const pr = Float64Array.from(prof);
  const trend = C.uniformFilter1d(pr, Math.max(5, Math.floor(pr.length / 6)), 'nearest');
  return { f, wp: pr.map((v, i) => v - trend[i]) };
}

const standardize = (a) => {
  let m = 0; for (const v of a) m += v; m /= a.length;
  const s = C.std(a) + 1e-12;
  return a.map((v) => (v - m) / s);
};

export function estimateDefocusBlind(g, rMin = 1.0, rMax = 25.0) {
  const { f, wp: w0 } = whitenedRadial(g);
  const wp = standardize(w0);
  let bestR = rMin, bestS = -Infinity;
  const nsteps = Math.round((rMax - rMin) / 0.1);
  for (let k = 0; k <= nsteps; k++) {
    const R = rMin + 0.1 * k;
    const m0 = Float64Array.from(f, (fi) => {
      const x = 2 * Math.PI * R * fi;
      return Math.log(Math.abs((2 * j1(x)) / Math.max(x, 1e-9)) + 0.02);
    });
    const tr = C.uniformFilter1d(m0, Math.max(5, Math.floor(m0.length / 6)), 'nearest');
    const m = standardize(m0.map((v, i) => v - tr[i]));
    let s = 0; for (let i = 0; i < m.length; i++) s += m[i] * wp[i];
    s /= m.length;
    if (s > bestS) { bestS = s; bestR = R; }
  }
  return { psf: diskPsf(bestR), radius: bestR, score: bestS };
}

export function estimateMotionBlind(g, maxLength = 60) {
  const ls = logSpectrum(g), n = ls.h, c = Math.floor(n / 2);
  const { prof } = radialProfile(ls);
  const sp = C.uniformFilter1d(prof, 5);
  const wl = img(n, n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const r = Math.hypot(y - c, x - c), i = y * n + x;
      wl.d[i] = r < 0.45 * n && r > 2 ? ls.d[i] - interp(r, null, sp) : 0;
    }
  const lo = c - Math.floor(n / 3), hi = c + Math.floor(n / 3);
  const projVar = (a) => {
    const rot = C.rotate(wl, -a), col = new Float64Array(hi - lo);
    for (let y = 0; y < n; y++) for (let x = lo; x < hi; x++) col[x - lo] += rot.d[y * n + x];
    const s = C.std(col); return s * s;
  };
  let a0 = 0, bv = -Infinity;
  for (let a = 0; a < 180; a += 2) { const v = projVar(a); if (v > bv) { bv = v; a0 = a; } }
  let bestA = a0; bv = -Infinity;
  for (let a = a0 - 2; a <= a0 + 2.001; a += 0.25) { const v = projVar(a); if (v > bv) { bv = v; bestA = a; } }
  const angle = ((bestA % 180) + 180) % 180;
  // cepstrum along the motion direction
  let m = Math.min(g.h, g.w); m -= m % 2;
  const gg = C.crop(g, 0, 0, m, m), mu = C.mean(gg), win = window2(m, m), x = img(m, m);
  for (let i = 0; i < m * m; i++) x.d[i] = (gg.d[i] - mu) * win.d[i];
  const G = C.rfft2(x.d, m, m), L = img(m, m);
  for (let i = 0; i < m * m; i++) L.d[i] = Math.log(Math.hypot(G.re[i], G.im[i]) + 1e-8);
  const cep = C.fftshift(img(m, m, C.irfft2({ re: L.d, im: new Float64Array(m * m) }, m, m)));
  const cc = Math.floor(m / 2), t = (angle * Math.PI) / 180;
  let bestL = 3, bestV = Infinity;
  for (let r = 2; r < Math.min(maxLength, cc - 1); r += 0.25) {
    if (r < 3) continue;
    const v = 0.5 * (C.bilinear(cep, cc - r * Math.sin(t), cc + r * Math.cos(t)) +
                     C.bilinear(cep, cc + r * Math.sin(t), cc - r * Math.cos(t)));
    if (v < bestV) { bestV = v; bestL = r; }
  }
  return { psf: motionPsf(bestL, angle), length: bestL, angle };
}

function spectrumFitScore(g, psf) {
  const ls = logSpectrum(g), n = ls.h, c = Math.floor(n / 2);
  const otf = C.psf2otf(psf, n, n), mm = img(n, n);
  for (let i = 0; i < n * n; i++) mm.d[i] = Math.log(Math.hypot(otf.re[i], otf.im[i]) + 0.02);
  const M = C.fftshift(mm);
  const t1 = C.uniformFilter1d(radialProfile(ls).prof, 9), t2 = C.uniformFilter1d(radialProfile(M).prof, 9);
  const a = [], b = [];
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const r = Math.hypot(y - c, x - c);
      if (r > 3 && r < 0.45 * n) { const i = y * n + x; a.push(ls.d[i] - interp(r, null, t1)); b.push(M.d[i] - interp(r, null, t2)); }
    }
  if (C.std(b) < 1e-6) return 0;
  const sa = standardize(a), sb = standardize(b);
  let s = 0; for (let i = 0; i < sa.length; i++) s += sa[i] * sb[i];
  return s / sa.length;
}

export function estimateAuto(g, motionThr = 0.28, diskThr = 0.55) {
  const mo = estimateMotionBlind(g), ms = spectrumFitScore(g, mo.psf);
  const de = estimateDefocusBlind(g);
  const scores = { motion: ms, disk: de.score };
  if (ms >= motionThr) return { psf: mo.psf, info: { kind: 'motion', length: mo.length, angle: mo.angle, scores } };
  if (de.score >= diskThr) return { psf: de.psf, info: { kind: 'disk', radius: de.radius, scores } };
  const ga = estimateGaussianBlind(g);
  return { psf: ga.psf, info: { kind: 'gaussian', sigma: ga.sigma, scores } };
}

// ========================================== 4. non-parametric blind kernel
function l0Latent(y, k, lam, betaMax = 1e5) {
  const { h, w } = y, N = h * w;
  const K = C.psf2otf(k, h, w), [Dx, Dy] = C.gradOtfs(h, w), Y = C.rfft2(y.d, h, w);
  const K2 = new Float64Array(N), D2 = new Float64Array(N), KtYr = new Float64Array(N), KtYi = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    K2[i] = K.re[i] ** 2 + K.im[i] ** 2;
    D2[i] = Dx.re[i] ** 2 + Dx.im[i] ** 2 + Dy.re[i] ** 2 + Dy.im[i] ** 2;
    KtYr[i] = K.re[i] * Y.re[i] + K.im[i] * Y.im[i];
    KtYi[i] = K.re[i] * Y.im[i] - K.im[i] * Y.re[i];
  }
  let x = Float64Array.from(y.d);
  const hh = new Float64Array(N), vv = new Float64Array(N);
  for (let beta = 2 * lam; beta < betaMax; beta *= 2) {
    for (let r = 0; r < h; r++)
      for (let c = 0; c < w; c++) {
        const i = r * w + c, a = x[r * w + ((c + 1) % w)] - x[i], b = x[((r + 1) % h) * w + c] - x[i];
        if (a * a + b * b < lam / beta) { hh[i] = 0; vv[i] = 0; } else { hh[i] = a; vv[i] = b; }
      }
    const H = C.rfft2(hh, h, w), V = C.rfft2(vv, h, w), re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const nr = KtYr[i] + beta * (Dx.re[i] * H.re[i] + Dx.im[i] * H.im[i] + Dy.re[i] * V.re[i] + Dy.im[i] * V.im[i]);
      const ni = KtYi[i] + beta * (Dx.re[i] * H.im[i] - Dx.im[i] * H.re[i] + Dy.re[i] * V.im[i] - Dy.im[i] * V.re[i]);
      const den = K2[i] + beta * D2[i];
      re[i] = nr / den; im[i] = ni / den;
    }
    x = C.irfft2({ re, im }, h, w);
  }
  return img(h, w, x);
}

function kernelStep(x, y, ks, gamma = 2.0) {
  const { h, w } = y, N = h * w, win = window2(h, w);
  const grads = (a) => {
    const gx = new Float64Array(N), gy = new Float64Array(N);
    for (let r = 0; r < h; r++)
      for (let c = 0; c < w; c++) {
        const i = r * w + c, s = Math.sqrt(win.d[i]);
        gx[i] = (a.d[r * w + ((c + 1) % w)] - a.d[i]) * s;
        gy[i] = (a.d[((r + 1) % h) * w + c] - a.d[i]) * s;
      }
    return [C.rfft2(gx, h, w), C.rfft2(gy, h, w)];
  };
  const xs = grads(x), ys = grads(y), re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let nr = 0, ni = 0, den = gamma;
    for (let k = 0; k < 2; k++) {
      const ar = xs[k].re[i], ai = xs[k].im[i], br = ys[k].re[i], bi = ys[k].im[i];
      nr += ar * br + ai * bi; ni += ar * bi - ai * br; den += ar * ar + ai * ai;
    }
    re[i] = nr / den; im[i] = ni / den;
  }
  const full = C.fftshift(img(h, w, C.irfft2({ re, im }, h, w)));
  const half = Math.floor(ks / 2), k = C.crop(full, Math.floor(h / 2) - half, Math.floor(w / 2) - half, ks, ks);
  let mx = 0;
  for (let i = 0; i < k.d.length; i++) { if (k.d[i] < 0) k.d[i] = 0; mx = Math.max(mx, k.d[i]); }
  if (mx <= 0) return null;
  for (let i = 0; i < k.d.length; i++) if (k.d[i] < 0.05 * mx) k.d[i] = 0;
  const { labels, n } = C.label(Array.from(k.d, (v) => v > 0), ks, ks);
  if (n > 1) {
    const mass = new Float64Array(n + 1);
    for (let i = 0; i < k.d.length; i++) mass[labels[i]] += k.d[i];
    let mm = 0; for (let l = 1; l <= n; l++) mm = Math.max(mm, mass[l]);
    for (let i = 0; i < k.d.length; i++) if (labels[i] && mass[labels[i]] < 0.1 * mm) k.d[i] = 0;
  }
  return centerPsf(k);
}

export function blindEstimate(g, size = 25, { lam = 4e-3, itersPerLevel = 5, gamma = 2.0, maxDim = 512, progress } = {}) {
  let y0 = g;
  if (Math.max(g.h, g.w) > maxDim) {
    const r0 = Math.max(0, Math.floor((g.h - maxDim) / 2)), c0 = Math.max(0, Math.floor((g.w - maxDim) / 2));
    y0 = C.crop(g, r0, c0, Math.min(maxDim, g.h), Math.min(maxDim, g.w));
  }
  size = odd(size);
  const ksizes = [size];
  while (ksizes[ksizes.length - 1] > 5) ksizes.push(odd(ksizes[ksizes.length - 1] / Math.SQRT2));
  ksizes.reverse();
  let k = null;
  ksizes.forEach((ks, li) => {
    const scale = ks / size, ys = scale < 1 ? C.zoom(y0, scale) : y0;
    if (!k) {
      k = img(ks, ks); const c = Math.floor(ks / 2);
      for (let d = -1; d <= 1; d++) { k.d[c * ks + c + d] = 1; k.d[(c + d) * ks + c] = 1; }
      k = C.normalizePsf(k);
    } else {
      let kz = C.zoom(k, ks / k.h);
      if (kz.h !== ks) { const t = img(ks, ks); for (let r = 0; r < Math.min(ks, kz.h); r++) for (let c = 0; c < Math.min(ks, kz.w); c++) t.d[r * ks + c] = kz.d[r * kz.w + c]; kz = t; }
      for (let i = 0; i < kz.d.length; i++) kz.d[i] = Math.max(kz.d[i], 0);
      k = centerPsf(kz);
    }
    const H = C.nextFastLen(ys.h + 2 * ks), W = C.nextFastLen(ys.w + 2 * ks);
    const yp = C.padReflect(ys, ks, H - ys.h - ks, ks, W - ys.w - ks);
    let l = lam;
    for (let it = 0; it < itersPerLevel; it++) {
      const x = l0Latent(yp, k, l), kn = kernelStep(x, yp, ks, gamma);
      if (kn) k = kn;
      l = Math.max(l / 1.1, 1e-4);
    }
    if (progress) progress((li + 1) / ksizes.length, `blind kernel: level ${li + 1}/${ksizes.length} (${ks}×${ks})`);
  });
  return C.normalizePsf(k);
}
