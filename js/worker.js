// Computation worker: all heavy maths runs here so the page stays responsive.
import * as C from './lib/core.js';
import * as P from './lib/psf.js';
import * as D from './lib/deconvolve.js';

const toImg = (o) => C.img(o.h, o.w, o.d);
const fmt = (v, d = 2) => Number(v).toFixed(d);

function seededNormal(seed) {
  let s = seed >>> 0;
  const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => Math.sqrt(-2 * Math.log(rnd() + 1e-300)) * Math.cos(2 * Math.PI * rnd());
}

function estimate(gray, source, prm, progress) {
  switch (source) {
    case 'auto': {
      progress(0.1, 'Analysing the spectrum (motion, defocus, Gaussian)…');
      const { psf, info } = P.estimateAuto(gray);
      const desc = info.kind === 'motion' ? `Auto → linear motion\nlength ${fmt(info.length)} px\nangle ${fmt(info.angle, 1)}°`
        : info.kind === 'disk' ? `Auto → defocus disk\nradius ${fmt(info.radius)} px`
        : `Auto → Gaussian\nσ = ${fmt(info.sigma)} px`;
      return { psf, desc, info };
    }
    case 'gaussian': return { psf: P.gaussianPsf(prm.sigma), desc: `Gaussian\nσ = ${fmt(prm.sigma)} px` };
    case 'disk': return { psf: P.diskPsf(prm.radius), desc: `Defocus disk\nradius ${fmt(prm.radius)} px` };
    case 'motion': return { psf: P.motionPsf(prm.length, prm.angle), desc: `Linear motion\nlength ${fmt(prm.length, 1)} px\nangle ${fmt(prm.angle, 0)}°` };
    case 'blind': {
      const psf = P.blindEstimate(gray, prm.ksize, { progress });
      return { psf, desc: `Blind kernel\n${psf.w}×${psf.h} px` };
    }
    case 'point': {
      const { psf, count } = P.estimateFromPointSource(gray, prm.box);
      return { psf, desc: `From point sources\n${count} point(s), ${psf.w}×${psf.h} px` };
    }
    case 'edge': {
      const { psf, sigma } = P.estimateFromEdge(gray);
      return { psf, desc: `From straight edge\nGaussian σ = ${fmt(sigma)} px` };
    }
    default: throw new Error('Unknown PSF source ' + source);
  }
}

self.onmessage = (ev) => {
  const { id, op } = ev.data;
  const progress = (f, msg) => self.postMessage({ id, progress: f, msg });
  try {
    if (op === 'estimate') {
      const r = estimate(toImg(ev.data.gray), ev.data.source, ev.data.params, progress);
      self.postMessage({ id, result: { psf: r.psf, desc: r.desc } }, [r.psf.d.buffer]);
    } else if (op === 'deconvolve') {
      const info = {};
      const out = D.deconvolve(toImg(ev.data.ch), toImg(ev.data.psf), ev.data.method,
        { ...ev.data.opts, info, progress: (f) => progress(f) });
      self.postMessage({ id, result: { out, param: info.param } }, [out.d.buffer]);
    } else if (op === 'simulate') {
      const ch = toImg(ev.data.ch), out = C.blur(ch, toImg(ev.data.psf)), nrm = seededNormal(ev.data.seed);
      for (let i = 0; i < out.d.length; i++) out.d[i] = Math.min(1, Math.max(0, out.d[i] + ev.data.noise * nrm()));
      self.postMessage({ id, result: { out } }, [out.d.buffer]);
    } else if (op === 'metrics') {
      const ref = ev.data.ref.map(toImg), x = ev.data.x.map(toImg);
      let mse = 0, s = 0;   // PSNR over all channels together (as the Python toolkit)
      ref.forEach((r, k) => { mse += 10 ** (-C.psnr(r, x[k]) / 10); s += C.ssim(r, x[k]); });
      self.postMessage({ id, result: { psnr: 10 * Math.log10(ref.length / mse), ssim: s / ref.length } });
    }
  } catch (e) {
    self.postMessage({ id, error: e.message || String(e) });
  }
};
