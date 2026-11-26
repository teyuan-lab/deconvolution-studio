// 2-D float image helpers that mirror the numpy / scipy.ndimage functions the
// Python toolkit uses.  An image is {h, w, d: Float64Array} (row-major).
import { fft2, rfft2, irfft2, nextFastLen } from './fft.js';

export { nextFastLen };

export const img = (h, w, d) => ({ h, w, d: d || new Float64Array(h * w) });
export const clone = (a) => img(a.h, a.w, Float64Array.from(a.d));

export function sum(a) { let s = 0; for (const v of a.d) s += v; return s; }
export function mean(a) { return sum(a) / a.d.length; }
export function max(a) { let m = -Infinity; for (const v of a.d) if (v > m) m = v; return m; }
export function std(arr) {
  let m = 0; for (const v of arr) m += v; m /= arr.length;
  let s = 0; for (const v of arr) s += (v - m) * (v - m);
  return Math.sqrt(s / arr.length);
}
export function median(arr) {
  const s = Float64Array.from(arr).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}
export function clip01(a) {
  const o = clone(a);
  for (let i = 0; i < o.d.length; i++) o.d[i] = Math.min(1, Math.max(0, o.d[i]));
  return o;
}

// ---------------------------------------------------------------- indexing
/** scipy.ndimage 'reflect' (half-sample symmetric): d c b a | a b c d */
export function reflectIdx(i, n) {
  if (n === 1) return 0;
  const p = 2 * n;
  i = ((i % p) + p) % p;
  return i < n ? i : p - 1 - i;
}
/** numpy.pad 'reflect' (whole-sample symmetric): d c b | a b c d */
export function mirrorIdx(i, n) {
  if (n === 1) return 0;
  const p = 2 * n - 2;
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
}
const nearestIdx = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);

export function crop(a, r0, c0, h, w) {
  const o = img(h, w);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) o.d[r * w + c] = a.d[(r0 + r) * a.w + c0 + c];
  return o;
}

/** numpy.pad(mode='reflect') */
export function padReflect(a, top, bottom, left, right) {
  const H = a.h + top + bottom, W = a.w + left + right, o = img(H, W);
  for (let r = 0; r < H; r++) {
    const sr = mirrorIdx(r - top, a.h);
    for (let c = 0; c < W; c++) o.d[r * W + c] = a.d[sr * a.w + mirrorIdx(c - left, a.w)];
  }
  return o;
}

export function hanning(n) {
  const w = new Float64Array(n);
  if (n === 1) { w[0] = 1; return w; }
  for (let k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (n - 1));
  return w;
}

// -------------------------------------------------------------- filtering
function conv1d(a, kernel, origin, axis, mode) {
  // correlate along axis with kernel centred at index `origin`
  const { h, w } = a, o = img(h, w), K = kernel.length;
  const idx = mode === 'nearest' ? nearestIdx : mode === 'mirror' ? mirrorIdx : reflectIdx;
  if (axis === 1) {
    for (let r = 0; r < h; r++) {
      const base = r * w;
      for (let c = 0; c < w; c++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += kernel[k] * a.d[base + idx(c + k - origin, w)];
        o.d[base + c] = s;
      }
    }
  } else {
    for (let r = 0; r < h; r++)
      for (let c = 0; c < w; c++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += kernel[k] * a.d[idx(r + k - origin, h) * w + c];
        o.d[r * w + c] = s;
      }
  }
  return o;
}

export function gaussianKernel(sigma, truncate = 4.0) {
  const rad = Math.floor(truncate * sigma + 0.5);
  const k = new Float64Array(2 * rad + 1);
  let s = 0;
  for (let i = -rad; i <= rad; i++) { k[i + rad] = Math.exp(-0.5 * (i * i) / (sigma * sigma)); s += k[i + rad]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

/** scipy.ndimage.gaussian_filter (mode='reflect') */
export function gaussianFilter(a, sigma, truncate = 4.0) {
  if (sigma <= 0) return clone(a);
  const k = gaussianKernel(sigma, truncate), rad = (k.length - 1) / 2;
  return conv1d(conv1d(a, k, rad, 1, 'reflect'), k, rad, 0, 'reflect');
}

/** scipy.ndimage.uniform_filter (both axes, mode='reflect') */
export function uniformFilter(a, size) {
  const k = new Float64Array(size).fill(1 / size), origin = Math.floor(size / 2);
  return conv1d(conv1d(a, k, origin, 1, 'reflect'), k, origin, 0, 'reflect');
}

/** scipy.ndimage.uniform_filter1d on a 1-D Float64Array */
export function uniformFilter1d(x, size, mode = 'reflect') {
  const n = x.length, o = new Float64Array(n), origin = Math.floor(size / 2);
  const idx = mode === 'nearest' ? nearestIdx : reflectIdx;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < size; k++) s += x[idx(i + k - origin, n)];
    o[i] = s / size;
  }
  return o;
}

/** scipy.ndimage.gaussian_filter1d on a 1-D array (mode='reflect') */
export function gaussianFilter1d(x, sigma) {
  const k = gaussianKernel(sigma), rad = (k.length - 1) / 2, n = x.length, o = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < k.length; j++) s += k[j] * x[reflectIdx(i + j - rad, n)];
    o[i] = s;
  }
  return o;
}

/** scipy.ndimage.maximum_filter (separable, mode='reflect') */
export function maximumFilter(a, size) {
  const origin = Math.floor(size / 2), { h, w } = a;
  const t = img(h, w), o = img(h, w);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      let m = -Infinity;
      for (let k = 0; k < size; k++) m = Math.max(m, a.d[r * w + reflectIdx(c + k - origin, w)]);
      t.d[r * w + c] = m;
    }
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      let m = -Infinity;
      for (let k = 0; k < size; k++) m = Math.max(m, t.d[reflectIdx(r + k - origin, h) * w + c]);
      o.d[r * w + c] = m;
    }
  return o;
}

/** Approximate median filter for large windows: exact median on a coarse
 *  grid (step = size/4), bilinearly interpolated back. */
export function medianFilterApprox(a, size) {
  const step = Math.max(1, Math.floor(size / 4)), h2 = Math.ceil(a.h / step), w2 = Math.ceil(a.w / step);
  const coarse = img(h2, w2), half = Math.floor(size / 2), buf = [];
  for (let i = 0; i < h2; i++)
    for (let j = 0; j < w2; j++) {
      const r = Math.min(a.h - 1, i * step), c = Math.min(a.w - 1, j * step);
      buf.length = 0;
      for (let dr = -half; dr <= half; dr += 2)
        for (let dc = -half; dc <= half; dc += 2)
          buf.push(a.d[reflectIdx(r + dr, a.h) * a.w + reflectIdx(c + dc, a.w)]);
      coarse.d[i * w2 + j] = median(buf);
    }
  const o = img(a.h, a.w);
  for (let r = 0; r < a.h; r++)
    for (let c = 0; c < a.w; c++) o.d[r * a.w + c] = bilinear(coarse, r / step, c / step, true);
  return o;
}

/** np.gradient -> [gy, gx] */
export function gradient(a) {
  const { h, w } = a, gy = img(h, w), gx = img(h, w);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      gx.d[i] = w === 1 ? 0 : c === 0 ? a.d[i + 1] - a.d[i] : c === w - 1 ? a.d[i] - a.d[i - 1] : 0.5 * (a.d[i + 1] - a.d[i - 1]);
      gy.d[i] = h === 1 ? 0 : r === 0 ? a.d[i + w] - a.d[i] : r === h - 1 ? a.d[i] - a.d[i - w] : 0.5 * (a.d[i + w] - a.d[i - w]);
    }
  return [gy, gx];
}

// ------------------------------------------------------------ geometry
/** bilinear sample at (row, col); outside -> 0 (or clamped if `clamp`) */
export function bilinear(a, y, x, clamp = false) {
  const { h, w, d } = a;
  if (clamp) { y = Math.min(Math.max(y, 0), h - 1); x = Math.min(Math.max(x, 0), w - 1); }
  const y0 = Math.floor(y), x0 = Math.floor(x), fy = y - y0, fx = x - x0;
  const g = (r, c) => (r < 0 || c < 0 || r >= h || c >= w ? 0 : d[r * w + c]);
  if (y < -1 || x < -1 || y > h || x > w) return 0;
  return (1 - fy) * ((1 - fx) * g(y0, x0) + fx * g(y0, x0 + 1)) + fy * ((1 - fx) * g(y0 + 1, x0) + fx * g(y0 + 1, x0 + 1));
}

/** scipy.ndimage.rotate(a, angle, reshape=False, order=1, cval=0) */
export function rotate(a, angleDeg) {
  const t = (angleDeg * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
  const { h, w } = a, o = img(h, w), cy = (h - 1) / 2, cx = (w - 1) / 2;
  for (let r = 0; r < h; r++)
    for (let col = 0; col < w; col++) {
      const dy = r - cy, dx = col - cx;
      const iy = cy + c * dy + s * dx, ix = cx - s * dy + c * dx;
      o.d[r * w + col] = (iy < 0 || ix < 0 || iy > h - 1 || ix > w - 1) ? 0 : bilinear(a, iy, ix);
    }
  return o;
}

/** scipy.ndimage.shift(a, [sy, sx], order=1, mode='constant') */
export function shift(a, sy, sx) {
  const o = img(a.h, a.w);
  for (let r = 0; r < a.h; r++)
    for (let c = 0; c < a.w; c++) {
      const y = r - sy, x = c - sx;
      o.d[r * a.w + c] = (y < 0 || x < 0 || y > a.h - 1 || x > a.w - 1) ? 0 : bilinear(a, y, x);
    }
  return o;
}

/** scipy.ndimage.zoom(a, scale, order=1) */
export function zoom(a, scale) {
  const H = Math.round(a.h * scale), W = Math.round(a.w * scale), o = img(H, W);
  const sy = H > 1 ? (a.h - 1) / (H - 1) : 0, sx = W > 1 ? (a.w - 1) / (W - 1) : 0;
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) o.d[r * W + c] = bilinear(a, r * sy, c * sx, true);
  return o;
}

export function centerOfMass(a) {
  let s = 0, sy = 0, sx = 0;
  for (let r = 0; r < a.h; r++)
    for (let c = 0; c < a.w; c++) { const v = a.d[r * a.w + c]; s += v; sy += v * r; sx += v * c; }
  return [sy / s, sx / s];
}

/** connected components, 8-connectivity; returns {labels: Int32Array, n} */
export function label(mask, h, w) {
  const lab = new Int32Array(h * w);
  let n = 0;
  const stack = [];
  for (let i = 0; i < h * w; i++) {
    if (!mask[i] || lab[i]) continue;
    n++; lab[i] = n; stack.push(i);
    while (stack.length) {
      const j = stack.pop(), r = (j / w) | 0, c = j % w;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
          const k = rr * w + cc;
          if (mask[k] && !lab[k]) { lab[k] = n; stack.push(k); }
        }
    }
  }
  return { labels: lab, n };
}

// --------------------------------------------------------------- FFT ops
export { fft2, rfft2, irfft2 };

/** MATLAB-style psf2otf: zero-pad to (H, W), centre at (0,0), FFT. */
export function psf2otf(psf, H, W) {
  const re = new Float64Array(H * W), im = new Float64Array(H * W);
  const oy = Math.floor(psf.h / 2), ox = Math.floor(psf.w / 2);
  for (let r = 0; r < psf.h; r++)
    for (let c = 0; c < psf.w; c++) {
      const rr = (((r - oy) % H) + H) % H, cc = (((c - ox) % W) + W) % W;
      re[rr * W + cc] += psf.d[r * psf.w + c];
    }
  fft2(re, im, H, W);
  return { re, im };
}

/** forward-difference operator OTFs (Dx, Dy) */
export function gradOtfs(H, W) {
  return [psf2otf(img(1, 2, Float64Array.from([1, -1])), H, W),
          psf2otf(img(2, 1, Float64Array.from([1, -1])), H, W)];
}

/** circular convolution with an OTF */
export function applyOtf(a, otf, conj = false) {
  const X = rfft2(a.d, a.h, a.w), n = a.d.length, s = conj ? -1 : 1;
  for (let i = 0; i < n; i++) {
    const xr = X.re[i], xi = X.im[i], hr = otf.re[i], hi = s * otf.im[i];
    X.re[i] = xr * hr - xi * hi; X.im[i] = xr * hi + xi * hr;
  }
  return img(a.h, a.w, irfft2(X, a.h, a.w));
}

export function fftshift(a) {
  const { h, w } = a, o = img(h, w), sy = Math.floor(h / 2), sx = Math.floor(w / 2);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) o.d[((r + sy) % h) * w + ((c + sx) % w)] = a.d[r * w + c];
  return o;
}

// ------------------------------------------------------------ PSF basics
export function normalizePsf(p) {
  const o = clone(p);
  let s = 0;
  for (let i = 0; i < o.d.length; i++) { if (o.d[i] < 0) o.d[i] = 0; s += o.d[i]; }
  if (s <= 0) throw new Error('PSF has no positive energy');
  for (let i = 0; i < o.d.length; i++) o.d[i] /= s;
  return o;
}

/** blur for simulation: true convolution, scipy 'reflect' boundary */
export function blur(a, psf) {
  const py = psf.h, px = psf.w;
  const H = nextFastLen(a.h + 2 * py), W = nextFastLen(a.w + 2 * px);
  const p = img(H, W);
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) p.d[r * W + c] = a.d[reflectIdx(r - py, a.h) * a.w + reflectIdx(c - px, a.w)];
  return crop(applyOtf(p, psf2otf(psf, H, W)), py, px, a.h, a.w);
}

/** Blend borders with a blurred copy so the image is ~periodic. */
export function edgetaper(a, psf) {
  const blurred = applyOtf(a, psf2otf(psf, a.h, a.w));
  const prof = (n, pw) => {
    const w = Math.max(2, Math.min(pw, Math.floor(n / 4))), p = new Float64Array(n).fill(1);
    for (let i = 0; i < w; i++) {
      const v = 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / w);
      p[i] = v; p[n - 1 - i] = v;
    }
    return p;
  };
  const ar = prof(a.h, psf.h), ac = prof(a.w, psf.w), o = img(a.h, a.w);
  for (let r = 0; r < a.h; r++)
    for (let c = 0; c < a.w; c++) {
      const al = ar[r] * ac[c], i = r * a.w + c;
      o.d[i] = al * a.d[i] + (1 - al) * blurred.d[i];
    }
  return o;
}

/** reflect-pad to fast FFT size + edgetaper; returns {x, r0, c0} */
export function prepare(a, psf) {
  const H = nextFastLen(a.h + 2 * psf.h), W = nextFastLen(a.w + 2 * psf.w);
  const p = padReflect(a, psf.h, H - a.h - psf.h, psf.w, W - a.w - psf.w);
  return { x: edgetaper(p, psf), r0: psf.h, c0: psf.w };
}

/** Donoho MAD noise estimate from the Haar HH band */
export function estimateNoiseSigma(a) {
  const h = a.h - (a.h % 2), w = a.w - (a.w % 2), hh = new Float64Array((h / 2) * (w / 2));
  let k = 0;
  for (let r = 0; r < h; r += 2)
    for (let c = 0; c < w; c += 2) {
      const i = r * a.w + c;
      hh[k++] = Math.abs((a.d[i] - a.d[i + 1] - a.d[i + a.w] + a.d[i + a.w + 1]) / 2);
    }
  return median(hh) / 0.6745;
}

// ---------------------------------------------------------------- metrics
export function psnr(ref, x) {
  let s = 0;
  for (let i = 0; i < ref.d.length; i++) { const e = ref.d[i] - x.d[i]; s += e * e; }
  const mse = s / ref.d.length;
  return mse === 0 ? Infinity : 10 * Math.log10(1 / mse);
}

export function ssim(ref, x) {
  const c1 = 1e-4, c2 = 9e-4, f = (a) => gaussianFilter(a, 1.5, 3.5);
  const mul = (a, b) => { const o = img(a.h, a.w); for (let i = 0; i < o.d.length; i++) o.d[i] = a.d[i] * b.d[i]; return o; };
  const mx = f(ref), my = f(x), sxx = f(mul(ref, ref)), syy = f(mul(x, x)), sxy = f(mul(ref, x));
  let s = 0, n = 0;
  for (let r = 5; r < ref.h - 5; r++)
    for (let c = 5; c < ref.w - 5; c++) {
      const i = r * ref.w + c, a = mx.d[i], b = my.d[i];
      const vx = sxx.d[i] - a * a, vy = syy.d[i] - b * b, cv = sxy.d[i] - a * b;
      s += ((2 * a * b + c1) * (2 * cv + c2)) / ((a * a + b * b + c1) * (vx + vy + c2));
      n++;
    }
  return s / n;
}

// -------------------------------------------------------------- colour
/** split an RGB(A) ImageData-like into channels; returns {channels, gray} */
export function luminance(chs) {
  if (chs.length === 1) return chs[0];
  const o = img(chs[0].h, chs[0].w);
  for (let i = 0; i < o.d.length; i++) o.d[i] = 0.299 * chs[0].d[i] + 0.587 * chs[1].d[i] + 0.114 * chs[2].d[i];
  return o;
}
