// Mixed-radix complex FFT (radix 4, 2, 3, 5 + generic), a port of the
// KISS FFT algorithm.  Works for any length; fastest for 2^a 3^b 5^c.
// Data are split real / imaginary Float64Arrays.

const plans = new Map();

function factorize(n) {
  const f = [];
  let p = 4;
  const floorSqrt = Math.floor(Math.sqrt(n));
  do {
    while (n % p) {
      switch (p) {
        case 4: p = 2; break;
        case 2: p = 3; break;
        default: p += 2;
      }
      if (p > floorSqrt) p = n;
    }
    n /= p;
    f.push(p, n);
  } while (n > 1);
  return f;
}

function getPlan(n) {
  let plan = plans.get(n);
  if (plan) return plan;
  const twr = new Float64Array(n), twi = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ph = (-2 * Math.PI * i) / n;
    twr[i] = Math.cos(ph);
    twi[i] = Math.sin(ph);
  }
  plan = {
    n, twr, twi, factors: factorize(n),
    sr: new Float64Array(64), si: new Float64Array(64), // generic scratch
    bufR: new Float64Array(n), bufI: new Float64Array(n),
    colR: new Float64Array(0), colI: new Float64Array(0),
  };
  plans.set(n, plan);
  return plan;
}

function bfly2(R, I, off, fstride, plan, m) {
  const { twr, twi } = plan;
  for (let k = 0; k < m; k++) {
    const a = off + k, b = a + m, t = k * fstride;
    const tr = R[b] * twr[t] - I[b] * twi[t];
    const ti = R[b] * twi[t] + I[b] * twr[t];
    R[b] = R[a] - tr; I[b] = I[a] - ti;
    R[a] += tr; I[a] += ti;
  }
}

function bfly4(R, I, off, fstride, plan, m) {
  const { twr, twi } = plan;
  for (let k = 0; k < m; k++) {
    const i0 = off + k, i1 = i0 + m, i2 = i1 + m, i3 = i2 + m;
    const t1 = k * fstride, t2 = 2 * t1, t3 = 3 * t1;
    const s0r = R[i1] * twr[t1] - I[i1] * twi[t1], s0i = R[i1] * twi[t1] + I[i1] * twr[t1];
    const s1r = R[i2] * twr[t2] - I[i2] * twi[t2], s1i = R[i2] * twi[t2] + I[i2] * twr[t2];
    const s2r = R[i3] * twr[t3] - I[i3] * twi[t3], s2i = R[i3] * twi[t3] + I[i3] * twr[t3];
    const s5r = R[i0] - s1r, s5i = I[i0] - s1i;
    let f0r = R[i0] + s1r, f0i = I[i0] + s1i;
    const s3r = s0r + s2r, s3i = s0i + s2i;
    const s4r = s0r - s2r, s4i = s0i - s2i;
    R[i2] = f0r - s3r; I[i2] = f0i - s3i;
    R[i0] = f0r + s3r; I[i0] = f0i + s3i;
    R[i1] = s5r + s4i; I[i1] = s5i - s4r;
    R[i3] = s5r - s4i; I[i3] = s5i + s4r;
  }
}

function bfly3(R, I, off, fstride, plan, m) {
  const { twr, twi } = plan;
  const epi3i = twi[fstride * m];
  for (let k = 0; k < m; k++) {
    const i0 = off + k, i1 = i0 + m, i2 = i1 + m;
    const t1 = k * fstride, t2 = 2 * t1;
    const s1r = R[i1] * twr[t1] - I[i1] * twi[t1], s1i = R[i1] * twi[t1] + I[i1] * twr[t1];
    const s2r = R[i2] * twr[t2] - I[i2] * twi[t2], s2i = R[i2] * twi[t2] + I[i2] * twr[t2];
    const s3r = s1r + s2r, s3i = s1i + s2i;
    let s0r = s1r - s2r, s0i = s1i - s2i;
    let f1r = R[i0] - s3r * 0.5, f1i = I[i0] - s3i * 0.5;
    s0r *= epi3i; s0i *= epi3i;
    R[i0] += s3r; I[i0] += s3i;
    R[i2] = f1r + s0i; I[i2] = f1i - s0r;
    R[i1] = f1r - s0i; I[i1] = f1i + s0r;
  }
}

function bflyGeneric(R, I, off, fstride, plan, m, p) {
  const { twr, twi, n } = plan;
  if (plan.sr.length < p) { plan.sr = new Float64Array(p); plan.si = new Float64Array(p); }
  const sr = plan.sr, si = plan.si;
  for (let u = 0; u < m; u++) {
    for (let q1 = 0, k = u; q1 < p; q1++, k += m) { sr[q1] = R[off + k]; si[q1] = I[off + k]; }
    for (let q1 = 0, k = u; q1 < p; q1++, k += m) {
      let twidx = 0;
      let accR = sr[0], accI = si[0];
      for (let q = 1; q < p; q++) {
        twidx += fstride * k;
        if (twidx >= n) twidx -= n;
        accR += sr[q] * twr[twidx] - si[q] * twi[twidx];
        accI += sr[q] * twi[twidx] + si[q] * twr[twidx];
      }
      R[off + k] = accR; I[off + k] = accI;
    }
  }
}

function work(oR, oI, oOff, iR, iI, iOff, fstride, istride, plan, fi) {
  const p = plan.factors[fi], m = plan.factors[fi + 1];
  if (m === 1) {
    for (let k = 0; k < p; k++) {
      const s = iOff + k * fstride * istride;
      oR[oOff + k] = iR[s]; oI[oOff + k] = iI[s];
    }
  } else {
    for (let q = 0; q < p; q++) {
      work(oR, oI, oOff + q * m, iR, iI, iOff + q * fstride * istride, fstride * p, istride, plan, fi + 2);
    }
  }
  switch (p) {
    case 2: bfly2(oR, oI, oOff, fstride, plan, m); break;
    case 3: bfly3(oR, oI, oOff, fstride, plan, m); break;
    case 4: bfly4(oR, oI, oOff, fstride, plan, m); break;
    default: bflyGeneric(oR, oI, oOff, fstride, plan, m, p);
  }
}

/** In-place forward 1-D FFT of re/im starting at `off` with length n. */
export function fft1(re, im, n, off = 0) {
  if (n === 1) return;
  const plan = getPlan(n);
  const bR = plan.bufR, bI = plan.bufI;
  for (let i = 0; i < n; i++) { bR[i] = re[off + i]; bI[i] = im[off + i]; }
  work(re, im, off, bR, bI, 0, 1, 1, plan, 0);
}

/** In-place 2-D FFT (h rows x w cols, row-major).  inverse=true divides by h*w. */
export function fft2(re, im, h, w, inverse = false) {
  const N = h * w;
  if (inverse) for (let i = 0; i < N; i++) im[i] = -im[i];
  for (let r = 0; r < h; r++) fft1(re, im, w, r * w);
  const plan = getPlan(h);
  if (plan.colR.length < h) { plan.colR = new Float64Array(h); plan.colI = new Float64Array(h); }
  const cR = plan.colR, cI = plan.colI;
  for (let c = 0; c < w; c++) {
    for (let r = 0, i = c; r < h; r++, i += w) { cR[r] = re[i]; cI[r] = im[i]; }
    fft1(cR, cI, h, 0);
    for (let r = 0, i = c; r < h; r++, i += w) { re[i] = cR[r]; im[i] = cI[r]; }
  }
  if (inverse) {
    const s = 1 / N;
    for (let i = 0; i < N; i++) { re[i] *= s; im[i] = -im[i] * s; }
  }
}

/** FFT of a real image -> {re, im} (new arrays). */
export function rfft2(x, h, w) {
  const re = Float64Array.from(x), im = new Float64Array(h * w);
  fft2(re, im, h, w, false);
  return { re, im };
}

/** Real part of the inverse FFT of {re, im} (does not modify the input). */
export function irfft2(X, h, w) {
  const re = Float64Array.from(X.re), im = Float64Array.from(X.im);
  fft2(re, im, h, w, true);
  return re;
}

/** Smallest m >= n whose prime factors are <= 11 (scipy.fft.next_fast_len). */
export function nextFastLen(n) {
  for (let m = Math.max(1, n); ; m++) {
    let k = m;
    for (const p of [2, 3, 5, 7, 11]) while (k % p === 0) k /= p;
    if (k === 1) return m;
  }
}
