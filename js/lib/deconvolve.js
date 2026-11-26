// Non-blind deconvolution - JavaScript port of deconv/deconvolve.py.
import * as C from './core.js';

const { img } = C;

/** Wrap a 2-D algorithm: padding (masked "unknown boundary" or edge taper) and crop. */
function run2d(fn, image, psf, masked, opts) {
  psf = C.normalizePsf(psf);
  let x, r0, c0, mask = null;
  if (masked) {
    const H = C.nextFastLen(image.h + 2 * psf.h), W = C.nextFastLen(image.w + 2 * psf.w);
    x = C.padReflect(image, psf.h, H - image.h - psf.h, psf.w, W - image.w - psf.w);
    r0 = psf.h; c0 = psf.w;
    mask = new Float64Array(H * W);
    for (let r = 0; r < image.h; r++) mask.fill(1, (r0 + r) * W + c0, (r0 + r) * W + c0 + image.w);
  } else {
    ({ x, r0, c0 } = C.prepare(image, psf));
  }
  return C.crop(fn(x, psf, mask, opts), r0, c0, image.h, image.w);
}

function absSq(o, i) { return o.re[i] * o.re[i] + o.im[i] * o.im[i]; }

/** GCV choice of the Tikhonov weight (gradient regulariser). */
export function gcvLambda(image, psf, regularizer = 'gradient') {
  const { h, w } = image, N = h * w, H = C.psf2otf(psf, h, w), Y = C.rfft2(image.d, h, w);
  const H2 = new Float64Array(N), R2 = new Float64Array(N), Y2 = new Float64Array(N);
  const [Dx, Dy] = C.gradOtfs(h, w);
  for (let i = 0; i < N; i++) {
    H2[i] = absSq(H, i); Y2[i] = absSq(Y, i);
    R2[i] = regularizer === 'gradient' ? absSq(Dx, i) + absSq(Dy, i) : 1;
  }
  let bestG = Infinity, bestL = 1e-3;
  for (let k = 0; k < 49; k++) {
    const lam = 10 ** (-6 + (6 * k) / 48);
    let num = 0, den = 0;
    for (let i = 0; i < N; i++) {
      const omf = (lam * R2[i]) / (H2[i] + lam * R2[i]);
      num += omf * omf * Y2[i]; den += omf;
    }
    const g = num / (den * den);
    if (g < bestG) { bestG = g; bestL = lam; }
  }
  return bestL;
}

function linearFilter(x, psf, regFn) {
  const { h, w } = x, N = h * w, H = C.psf2otf(psf, h, w), Y = C.rfft2(x.d, h, w);
  const [Dx, Dy] = C.gradOtfs(h, w), re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const den = absSq(H, i) + regFn(absSq(Dx, i) + absSq(Dy, i));
    re[i] = (H.re[i] * Y.re[i] + H.im[i] * Y.im[i]) / den;
    im[i] = (H.re[i] * Y.im[i] - H.im[i] * Y.re[i]) / den;
  }
  return img(h, w, C.irfft2({ re, im }, h, w));
}

// `scale` multiplies the automatic weight; `info.param` reports the value used.
export function wiener(image, psf, { nsr = null, scale = 1, info = {} } = {}) {
  return run2d((x, p) => {
    const k = (nsr == null ? Math.max(60 * C.estimateNoiseSigma(x) ** 2, 1e-4) : nsr) * scale;
    info.param = `NSR = ${k.toExponential(2)}`;
    return linearFilter(x, p, () => k);
  }, image, psf, false);
}

export function tikhonov(image, psf, { lam = null, scale = 1, info = {} } = {}) {
  return run2d((x, p) => {
    const l = (lam == null ? gcvLambda(x, p) : lam) * scale;
    info.param = `λ = ${l.toExponential(2)} (GCV)`;
    return linearFilter(x, p, (d2) => l * d2);
  }, image, psf, false);
}

const grad = (x, h, w) => {
  const gx = new Float64Array(h * w), gy = new Float64Array(h * w);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      gx[i] = x[r * w + ((c + 1) % w)] - x[i];
      gy[i] = x[((r + 1) % h) * w + c] - x[i];
    }
  return [gx, gy];
};
const div = (px, py, h, w) => {
  const o = new Float64Array(h * w);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      o[i] = px[i] - px[r * w + ((c - 1 + w) % w)] + py[i] - py[((r - 1 + h) % h) * w + c];
    }
  return o;
};

export function richardsonLucy(image, psf, { iterations = 30, tv = 0, info = {}, progress } = {}) {
  info.param = tv > 0 ? `${iterations} iterations, TV ${tv.toExponential(1)}` : `${iterations} iterations`;
  return run2d((y0, p, mask) => {
    const { h, w } = y0, N = h * w, eps = 1e-7, H = C.psf2otf(p, h, w);
    const y = Float64Array.from(y0.d, (v) => Math.max(v, 0) + eps);
    const conv = (a, conj) => C.applyOtf(img(h, w, a), H, conj).d;
    const norm = conv(mask, true).map((v) => Math.max(v, 1e-3));
    let s = 0, cnt = 0;
    for (let i = 0; i < N; i++) if (mask[i] > 0) { s += y[i]; cnt++; }
    let x = new Float64Array(N).fill(s / cnt);
    const ratio = new Float64Array(N);
    for (let k = 0; k < iterations; k++) {
      const Hx = conv(x, false);
      for (let i = 0; i < N; i++) ratio[i] = (mask[i] * y[i]) / Math.max(Hx[i], eps);
      const corr = conv(ratio, true);
      if (tv > 0) {
        const [gx, gy] = grad(x, h, w);
        for (let i = 0; i < N; i++) { const m = Math.sqrt(gx[i] ** 2 + gy[i] ** 2 + 1e-6); gx[i] /= m; gy[i] /= m; }
        const dv = div(gx, gy, h, w);
        for (let i = 0; i < N; i++) x[i] = Math.max((x[i] * corr[i]) / norm[i] / Math.max(1 - tv * dv[i], 0.1), 0);
      } else {
        for (let i = 0; i < N; i++) x[i] = Math.max((x[i] * corr[i]) / norm[i], 0);
      }
      if (progress && (k % 5 === 4 || k === iterations - 1)) progress((k + 1) / iterations);
    }
    return img(h, w, x);
  }, image, psf, true);
}

export function tvAdmm(image, psf, { lam = null, rho = null, iterations = 80, mu = 1.0, scale = 1, info = {}, progress } = {}) {
  return run2d((y, p, mask) => {
    const { h, w } = y, N = h * w;
    if (lam == null) lam = 0.25 * gcvLambda(C.edgetaper(y, p), p) * scale;
    if (rho == null) rho = Math.max(10 * lam, 0.05);
    info.param = `λ = ${lam.toExponential(2)}`;
    const H = C.psf2otf(p, h, w), [Dx, Dy] = C.gradOtfs(h, w), den = new Float64Array(N);
    for (let i = 0; i < N; i++) den[i] = mu * absSq(H, i) + rho * (absSq(Dx, i) + absSq(Dy, i));
    const My = Float64Array.from(y.d, (v, i) => v * mask[i]);
    const x = Float64Array.from(y.d);
    const Hx = C.applyOtf(img(h, w, x), H).d;
    const v = Float64Array.from(Hx);
    const u1 = new Float64Array(N), zx = new Float64Array(N), zy = new Float64Array(N);
    const ux = new Float64Array(N), uy = new Float64Array(N);
    const zr = new Float64Array(N), zi = new Float64Array(N), b = new Float64Array(N), c = new Float64Array(N);
    const gx = new Float64Array(N), gy = new Float64Array(N);
    // mirror index N-k for the packed-real-FFT trick
    const mir = new Int32Array(N);
    for (let r = 0; r < h; r++) for (let cc = 0; cc < w; cc++) mir[r * w + cc] = ((h - r) % h) * w + ((w - cc) % w);
    const thr = lam / rho;
    for (let it = 0; it < iterations; it++) {
      // rhs = mu conj(H) FFT(v-u1) + rho FFT(Dx^T b + Dy^T c),  b = zx-ux, c = zy-uy
      for (let i = 0; i < N; i++) { b[i] = zx[i] - ux[i]; c[i] = zy[i] - uy[i]; }
      for (let r = 0; r < h; r++)
        for (let cc = 0; cc < w; cc++) {
          const i = r * w + cc;
          zr[i] = v[i] - u1[i];                                   // real part  -> A
          zi[i] = b[r * w + ((cc - 1 + w) % w)] - b[i] + c[((r - 1 + h) % h) * w + cc] - c[i]; // imag -> G
        }
      C.fft2(zr, zi, h, w);   // one complex FFT carries both real transforms
      const qr = new Float64Array(N), qi = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const j = mir[i];
        const Ar = 0.5 * (zr[i] + zr[j]), Ai = 0.5 * (zi[i] - zi[j]);   // A = (Z + conj Z~)/2
        const Gr = 0.5 * (zi[i] + zi[j]), Gi = -0.5 * (zr[i] - zr[j]);  // G = (Z - conj Z~)/(2i)
        const nr = (mu * (H.re[i] * Ar + H.im[i] * Ai) + rho * Gr) / den[i];
        const ni = (mu * (H.re[i] * Ai - H.im[i] * Ar) + rho * Gi) / den[i];
        // pack X and H*X so one inverse FFT returns x (real) and Hx (imag)
        const hr = H.re[i] * nr - H.im[i] * ni, hi = H.re[i] * ni + H.im[i] * nr;
        qr[i] = nr - hi; qi[i] = ni + hr;
      }
      C.fft2(qr, qi, h, w, true);
      for (let i = 0; i < N; i++) { x[i] = qr[i]; Hx[i] = qi[i]; }
      for (let i = 0; i < N; i++) v[i] = (My[i] + mu * (Hx[i] + u1[i])) / (mask[i] + mu);
      for (let r = 0; r < h; r++)
        for (let cc = 0; cc < w; cc++) {
          const i = r * w + cc;
          gx[i] = x[r * w + ((cc + 1) % w)] - x[i];
          gy[i] = x[((r + 1) % h) * w + cc] - x[i];
        }
      for (let i = 0; i < N; i++) {
        const vx = gx[i] + ux[i], vy = gy[i] + uy[i], m = Math.sqrt(vx * vx + vy * vy);
        const sh = Math.max(1 - thr / Math.max(m, 1e-12), 0);
        zx[i] = sh * vx; zy[i] = sh * vy;
        u1[i] += Hx[i] - v[i]; ux[i] += gx[i] - zx[i]; uy[i] += gy[i] - zy[i];
      }
      if (progress && (it % 5 === 4 || it === iterations - 1)) progress((it + 1) / iterations);
    }
    return img(h, w, x);
  }, image, psf, true);
}

export function deconvolve(image, psf, method = 'tv', opts = {}) {
  let out;
  if (method === 'wiener') out = wiener(image, psf, opts);
  else if (method === 'tikhonov') out = tikhonov(image, psf, opts);
  else if (method === 'rl') out = richardsonLucy(image, psf, opts);
  else if (method === 'rl_tv') out = richardsonLucy(image, psf, { iterations: 50, ...opts, tv: 0.003 * (opts.scale || 1) });
  else if (method === 'tv') out = tvAdmm(image, psf, opts);
  else throw new Error('unknown method ' + method);
  return C.clip01(out);
}
