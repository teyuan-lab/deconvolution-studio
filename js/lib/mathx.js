// Special functions and a small bounded least-squares solver.

/** Bessel J1 (Numerical Recipes bessj1, |err| < 1e-8) */
export function j1(x) {
  const ax = Math.abs(x);
  if (ax < 8.0) {
    const y = x * x;
    const a1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
    const a2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
    return a1 / a2;
  }
  const z = 8.0 / ax, y = z * z, xx = ax - 2.356194491;
  const a1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
  const a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  const ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * a1 - z * Math.sin(xx) * a2);
  return x < 0 ? -ans : ans;
}

/** erf via Numerical Recipes erfc Chebyshev fit (|rel err| < 1.2e-7) */
export function erf(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}

/** log(exp(a) + exp(b)) */
export const logaddexp = (a, b) => { const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); };

function solve(A, b) { // Gaussian elimination with partial pivoting (small systems)
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-300) return null;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/**
 * Bounded nonlinear least squares (Levenberg-Marquardt with projection onto
 * the box).  residual(q) -> Array of residuals.  Returns {x, cost} with
 * cost = 0.5 * sum r^2 (same definition as scipy.optimize.least_squares).
 */
export function leastSquares(residual, q0, lo, hi, maxIter = 200) {
  const n = q0.length;
  const clampQ = (q) => q.map((v, i) => Math.min(hi[i], Math.max(lo[i], v)));
  let q = clampQ(q0.slice());
  let r = residual(q), cost = 0.5 * r.reduce((s, v) => s + v * v, 0);
  let lambda = 1e-3;
  for (let it = 0; it < maxIter; it++) {
    // numerical Jacobian (forward differences)
    const J = [];
    for (let j = 0; j < n; j++) {
      const h = 1e-6 * Math.max(1, Math.abs(q[j]));
      const qp = q.slice(); qp[j] = Math.min(hi[j], qp[j] + h);
      let hh = qp[j] - q[j];
      if (hh === 0) { qp[j] = q[j] - h; hh = -h; }
      const rp = residual(qp);
      J.push(rp.map((v, i) => (v - r[i]) / hh));
    }
    const JTJ = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => J[a].reduce((s, v, i) => s + v * J[b][i], 0)));
    const JTr = J.map((col) => col.reduce((s, v, i) => s + v * r[i], 0));
    let improved = false;
    for (let tries = 0; tries < 12; tries++) {
      const A = JTJ.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lambda) + 1e-12 : v)));
      const step = solve(A, JTr.map((v) => -v));
      if (!step) { lambda *= 10; continue; }
      const qn = clampQ(q.map((v, i) => v + step[i]));
      const rn = residual(qn), cn = 0.5 * rn.reduce((s, v) => s + v * v, 0);
      if (Number.isFinite(cn) && cn < cost) {
        const rel = (cost - cn) / Math.max(cost, 1e-300);
        q = qn; r = rn; cost = cn; lambda = Math.max(lambda / 3, 1e-12); improved = true;
        if (rel < 1e-12) return { x: q, cost };
        break;
      }
      lambda *= 4;
    }
    if (!improved) break;
  }
  return { x: q, cost };
}
