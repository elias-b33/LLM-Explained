/**
 * WGSL compute shaders for the WebGPU training engine. Each one mirrors a
 * kernel of the CPU engine in ../gpt.ts (same maths, same memory layout), so
 * the two can be compared number by number (see selftest.ts).
 *
 * Conventions: binding 0 is always a uniform block `u`; every other binding is
 * a storage buffer. Parameters and gradients live in two big flat buffers and
 * kernels address them through element offsets in `u`. Elementwise kernels run
 * 256 threads per workgroup over a 2D grid, so very long arrays never exceed
 * the 65535-workgroups-per-dimension limit.
 */

const FLAT_INDEX = `let i = gid.x + gid.y * nw.x * 256u;`
const FLAT_ARGS = `@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u`

/** x0[n,c] = tok[idx[n], c] + pos[n % T, c] */
const embed = /* wgsl */ `
struct U { N: u32, T: u32, d: u32, tokOff: u32, posOff: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.N * u.d) { return; }
  let n = i / u.d;
  let c = i % u.d;
  let t = n % u.T;
  outp[i] = P[u.tokOff + idx[n] * u.d + c] + P[u.posOff + t * u.d + c];
}`

/** Layer norm, one workgroup per row. Two-pass mean/variance like the CPU engine. */
const layernorm = /* wgsl */ `
struct U { N: u32, C: u32, gOff: u32, bOff: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read_write> mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let row = wg.x;
  if (row >= u.N) { return; }
  let off = row * u.C;
  var s = 0.0;
  for (var i = li; i < u.C; i += 64u) { s += inp[off + i]; }
  red[li] = s;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  let m = red[0] / f32(u.C);
  workgroupBarrier();
  var v = 0.0;
  for (var i = li; i < u.C; i += 64u) { let x = inp[off + i] - m; v += x * x; }
  red[li] = v;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  let rs = 1.0 / sqrt(red[0] / f32(u.C) + 1e-5);
  for (var i = li; i < u.C; i += 64u) {
    outp[off + i] = (inp[off + i] - m) * rs * P[u.gOff + i] + P[u.bOff + i];
  }
  if (li == 0u) { mean[row] = m; rstd[row] = rs; }
}`

/** Layer norm backward w.r.t. the input, one workgroup per row. Accumulates into dinp unless acc == 0. */
const layernormBackInput = /* wgsl */ `
struct U { N: u32, C: u32, gOff: u32, acc: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<f32>;
@group(0) @binding(4) var<storage, read> mean: array<f32>;
@group(0) @binding(5) var<storage, read> rstd: array<f32>;
@group(0) @binding(6) var<storage, read_write> dinp: array<f32>;
var<workgroup> r1: array<f32, 64>;
var<workgroup> r2: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let row = wg.x;
  if (row >= u.N) { return; }
  let off = row * u.C;
  let m = mean[row];
  let s = rstd[row];
  var a = 0.0;
  var b = 0.0;
  for (var i = li; i < u.C; i += 64u) {
    let norm = (inp[off + i] - m) * s;
    let dn = P[u.gOff + i] * dout[off + i];
    a += dn;
    b += dn * norm;
  }
  r1[li] = a;
  r2[li] = b;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { r1[li] += r1[li + st]; r2[li] += r2[li + st]; }
    workgroupBarrier();
  }
  let dm = r1[0] / f32(u.C);
  let dnm = r2[0] / f32(u.C);
  for (var i = li; i < u.C; i += 64u) {
    let norm = (inp[off + i] - m) * s;
    let dn = P[u.gOff + i] * dout[off + i];
    let val = (dn - dm - norm * dnm) * s;
    if (u.acc != 0u) { dinp[off + i] += val; } else { dinp[off + i] = val; }
  }
}`

/**
 * Column reductions use a 16x16 workgroup: 16 adjacent columns (coalesced
 * reads) x 16 row lanes, reduced through shared memory. Deterministic order.
 */
/** Layer norm backward w.r.t. gain and bias: G[g] += sum_n norm*dout, G[b] += sum_n dout. */
const layernormBackParams = /* wgsl */ `
struct U { N: u32, C: u32, gOff: u32, bOff: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> G: array<f32>;
var<workgroup> sg: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let c = wg.x * 16u + lid.x;
  var dg = 0.0;
  var db = 0.0;
  if (c < u.C) {
    for (var n = lid.y; n < u.N; n += 16u) {
      let d = dout[n * u.C + c];
      let norm = (inp[n * u.C + c] - mean[n]) * rstd[n];
      dg += norm * d;
      db += d;
    }
  }
  sg[lid.y * 16u + lid.x] = dg;
  sb[lid.y * 16u + lid.x] = db;
  workgroupBarrier();
  if (lid.y == 0u && c < u.C) {
    var tg = 0.0;
    var tb = 0.0;
    for (var y = 0u; y < 16u; y++) { tg += sg[y * 16u + lid.x]; tb += sb[y * 16u + lid.x]; }
    G[u.gOff + c] += tg;
    G[u.bOff + c] += tb;
  }
}`

/** Bias gradient: G[off + c] += sum_n D[n, c]. */
const colsum = /* wgsl */ `
struct U { N: u32, C: u32, off: u32, pad: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> D: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
var<workgroup> sd: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let c = wg.x * 16u + lid.x;
  var s = 0.0;
  if (c < u.C) {
    for (var n = lid.y; n < u.N; n += 16u) { s += D[n * u.C + c]; }
  }
  sd[lid.y * 16u + lid.x] = s;
  workgroupBarrier();
  if (lid.y == 0u && c < u.C) {
    var t = 0.0;
    for (var y = 0u; y < 16u; y++) { t += sd[y * 16u + lid.x]; }
    G[u.off + c] += t;
  }
}`

/**
 * Tiled matmul: C[M,N] (+)= opA(A)[M,K] * opB(B)[K,N] (+ bias[N]).
 * 64x64 output tile per workgroup, 16-deep K slices staged in shared memory,
 * each of the 256 threads computes a 4x4 block (strided by 16, so the shared
 * reads are bank-conflict free).
 *   transA: A is stored [K,M] (used for weight gradients: inp^T * dout)
 *   transB: B is stored [N,K] (used for input gradients: dout * W^T)
 * flags: bit 0 = add bias, bit 1 = accumulate into C.
 */
function matmul(transA: boolean, transB: boolean): string {
  const loadA = transA
    ? `let kk = idx / 64u; let mm = idx % 64u; let gr = row0 + mm; let gk = k0 + kk;
       var v = 0.0; if (gr < u.M && gk < u.K) { v = A[u.aOff + gk * u.M + gr]; } As[kk * 64u + mm] = v;`
    : `let mm = idx / 16u; let kk = idx % 16u; let gr = row0 + mm; let gk = k0 + kk;
       var v = 0.0; if (gr < u.M && gk < u.K) { v = A[u.aOff + gr * u.K + gk]; } As[kk * 64u + mm] = v;`
  const loadB = transB
    ? `let nn = idx / 16u; let kk = idx % 16u; let gc = col0 + nn; let gk = k0 + kk;
       var v = 0.0; if (gc < u.N && gk < u.K) { v = B[u.bOff + gc * u.K + gk]; } Bs[kk * 64u + nn] = v;`
    : `let kk = idx / 64u; let nn = idx % 64u; let gc = col0 + nn; let gk = k0 + kk;
       var v = 0.0; if (gc < u.N && gk < u.K) { v = B[u.bOff + gk * u.N + gc]; } Bs[kk * 64u + nn] = v;`
  return /* wgsl */ `
struct U { M: u32, N: u32, K: u32, aOff: u32, bOff: u32, cOff: u32, biasOff: u32, flags: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> Bias: array<f32>;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * 64u;
  let col0 = wg.x * 64u;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < u.K; k0 += 16u) {
    for (var e = 0u; e < 4u; e++) { let idx = li + e * 256u; ${loadA} }
    for (var e = 0u; e < 4u; e++) { let idx = li + e * 256u; ${loadB} }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let a0 = As[kk * 64u + lid.y];
      let a1 = As[kk * 64u + lid.y + 16u];
      let a2 = As[kk * 64u + lid.y + 32u];
      let a3 = As[kk * 64u + lid.y + 48u];
      let b0 = Bs[kk * 64u + lid.x];
      let b1 = Bs[kk * 64u + lid.x + 16u];
      let b2 = Bs[kk * 64u + lid.x + 32u];
      let b3 = Bs[kk * 64u + lid.x + 48u];
      acc[0] += a0 * b0; acc[1] += a0 * b1; acc[2] += a0 * b2; acc[3] += a0 * b3;
      acc[4] += a1 * b0; acc[5] += a1 * b1; acc[6] += a1 * b2; acc[7] += a1 * b3;
      acc[8] += a2 * b0; acc[9] += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
      acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = row0 + lid.y + 16u * i;
    if (r >= u.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let c = col0 + lid.x + 16u * j;
      if (c >= u.N) { continue; }
      var v = acc[i * 4u + j];
      if ((u.flags & 1u) != 0u) { v += Bias[u.biasOff + c]; }
      let o = u.cOff + r * u.N + c;
      if ((u.flags & 2u) != 0u) { v += C[o]; }
      C[o] = v;
    }
  }
}`
}

/** y += x */
const add = /* wgsl */ `
struct U { n: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.n) { return; }
  y[i] += x[i];
}`

const GELU = /* wgsl */ `
const GELU_K = 0.7978845608028654;
fn tanh_s(x: f32) -> f32 { return tanh(clamp(x, -10.0, 10.0)); }`

const geluFwd = /* wgsl */ `
${GELU}
struct U { n: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.n) { return; }
  let x = inp[i];
  outp[i] = 0.5 * x * (1.0 + tanh_s(GELU_K * (x + 0.044715 * x * x * x)));
}`

const geluBack = /* wgsl */ `
${GELU}
struct U { n: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> dout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dinp: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.n) { return; }
  let x = inp[i];
  let th = tanh_s(GELU_K * (x + 0.044715 * x * x * x));
  let du = GELU_K * (1.0 + 3.0 * 0.044715 * x * x);
  dinp[i] = (0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du) * dout[i];
}`

const ATT_HEAD = /* wgsl */ `
struct U { B: u32, T: u32, d: u32, H: u32, scale: f32 }
@group(0) @binding(0) var<uniform> u: U;`

/** One thread per (batch, head, query position t): scores, softmax, weighted sum of values. */
const attnFwd = /* wgsl */ `
${ATT_HEAD}
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> att: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= u.B * u.H * u.T) { return; }
  let t = r % u.T;
  let bh = r / u.T;
  let h = bh % u.H;
  let b = bh / u.H;
  let hs = u.d / u.H;
  let C3 = 3u * u.d;
  let qOff = (b * u.T + t) * C3 + h * hs;
  let aOff = r * u.T;
  var mx = -3.4e38;
  for (var t2 = 0u; t2 <= t; t2++) {
    let kOff = (b * u.T + t2) * C3 + u.d + h * hs;
    var s = 0.0;
    for (var i = 0u; i < hs; i++) { s += qkv[qOff + i] * qkv[kOff + i]; }
    s *= u.scale;
    att[aOff + t2] = s;
    mx = max(mx, s);
  }
  var sum = 0.0;
  for (var t2 = 0u; t2 <= t; t2++) {
    let e = exp(att[aOff + t2] - mx);
    att[aOff + t2] = e;
    sum += e;
  }
  let inv = 1.0 / sum;
  for (var t2 = 0u; t2 <= t; t2++) { att[aOff + t2] *= inv; }
  for (var t2 = t + 1u; t2 < u.T; t2++) { att[aOff + t2] = 0.0; }
  let oOff = (b * u.T + t) * u.d + h * hs;
  for (var i = 0u; i < hs; i++) {
    var acc = 0.0;
    for (var t2 = 0u; t2 <= t; t2++) {
      acc += att[aOff + t2] * qkv[(b * u.T + t2) * C3 + 2u * u.d + h * hs + i];
    }
    outp[oOff + i] = acc;
  }
}`

/**
 * Attention backward, part 1: one thread per query row (b, h, t).
 * datt[t, t2] = p * (dp - sum(p * dp)) * scale  (gradient w.r.t. the raw score)
 * dq[t] = sum_t2 datt[t, t2] * k[t2]
 * The CPU engine scatters into dk/dv inside the same loop; on the GPU that would
 * need atomics, so dk/dv are gathered per key row in part 2 instead.
 */
const attnBack1 = /* wgsl */ `
${ATT_HEAD}
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> att: array<f32>;
@group(0) @binding(3) var<storage, read> dout: array<f32>;
@group(0) @binding(4) var<storage, read_write> datt: array<f32>;
@group(0) @binding(5) var<storage, read_write> dqkv: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= u.B * u.H * u.T) { return; }
  let t = r % u.T;
  let bh = r / u.T;
  let h = bh % u.H;
  let b = bh / u.H;
  let hs = u.d / u.H;
  let C3 = 3u * u.d;
  let aOff = r * u.T;
  let dOff = (b * u.T + t) * u.d + h * hs;
  var dotSum = 0.0;
  for (var t2 = 0u; t2 <= t; t2++) {
    let vOff = (b * u.T + t2) * C3 + 2u * u.d + h * hs;
    var dp = 0.0;
    for (var i = 0u; i < hs; i++) { dp += dout[dOff + i] * qkv[vOff + i]; }
    datt[aOff + t2] = dp;
    dotSum += att[aOff + t2] * dp;
  }
  for (var t2 = 0u; t2 <= t; t2++) {
    datt[aOff + t2] = att[aOff + t2] * (datt[aOff + t2] - dotSum) * u.scale;
  }
  for (var t2 = t + 1u; t2 < u.T; t2++) { datt[aOff + t2] = 0.0; }
  let qOff = (b * u.T + t) * C3 + h * hs;
  for (var i = 0u; i < hs; i++) {
    var acc = 0.0;
    for (var t2 = 0u; t2 <= t; t2++) {
      acc += datt[aOff + t2] * qkv[(b * u.T + t2) * C3 + u.d + h * hs + i];
    }
    dqkv[qOff + i] = acc;
  }
}`

/** Attention backward, part 2: one thread per key row (b, h, t2). dk = sum_t datt*q, dv = sum_t p*dout. */
const attnBack2 = /* wgsl */ `
${ATT_HEAD}
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> att: array<f32>;
@group(0) @binding(3) var<storage, read> datt: array<f32>;
@group(0) @binding(4) var<storage, read> dout: array<f32>;
@group(0) @binding(5) var<storage, read_write> dqkv: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= u.B * u.H * u.T) { return; }
  let t2 = r % u.T;
  let bh = r / u.T;
  let h = bh % u.H;
  let b = bh / u.H;
  let hs = u.d / u.H;
  let C3 = 3u * u.d;
  let rowBase = bh * u.T * u.T;
  let kOff = (b * u.T + t2) * C3 + u.d + h * hs;
  let vOff = kOff + u.d;
  for (var i = 0u; i < hs; i++) {
    var dk = 0.0;
    var dv = 0.0;
    for (var t = t2; t < u.T; t++) {
      let a = rowBase + t * u.T + t2;
      dk += datt[a] * qkv[(b * u.T + t) * C3 + h * hs + i];
      dv += att[a] * dout[(b * u.T + t) * u.d + h * hs + i];
    }
    dqkv[kOff + i] = dk;
    dqkv[vOff + i] = dv;
  }
}`

/** Softmax cross-entropy per row: rowLoss[n] and dlogits[n,:] = (softmax - onehot) / N. */
const xent = /* wgsl */ `
struct U { N: u32, V: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(4) var<storage, read_write> rowLoss: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = gid.x;
  if (n >= u.N) { return; }
  let off = n * u.V;
  var mx = -3.4e38;
  for (var i = 0u; i < u.V; i++) { mx = max(mx, logits[off + i]); }
  var sum = 0.0;
  for (var i = 0u; i < u.V; i++) { sum += exp(logits[off + i] - mx); }
  let tgt = targets[n];
  rowLoss[n] = log(sum) - (logits[off + tgt] - mx);
  let invN = 1.0 / f32(u.N);
  let invS = 1.0 / sum;
  for (var i = 0u; i < u.V; i++) {
    var p = exp(logits[off + i] - mx) * invS;
    if (i == tgt) { p -= 1.0; }
    dlogits[off + i] = p * invN;
  }
}`

/** Single workgroup: outp[slot] = mean(inp[0..n)). */
const meanReduce = /* wgsl */ `
struct U { n: u32, slot: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var s = 0.0;
  for (var i = li; i < u.n; i += 256u) { s += inp[i]; }
  red[li] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  if (li == 0u) { outp[u.slot] = red[0] / f32(u.n); }
}`

/** Gradient norm, pass 1: 256 workgroups each write the sum of squares of a strided slice. */
const normPartial = /* wgsl */ `
struct U { n: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  var s = 0.0;
  for (var i = wg.x * 256u + li; i < u.n; i += 65536u) { let g = G[i]; s += g * g; }
  red[li] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  if (li == 0u) { partial[wg.x] = red[0]; }
}`

/** Gradient norm, pass 2: normSq[0] = sum of the partials. */
const normFinal = /* wgsl */ `
struct U { n: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> normSq: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var s = 0.0;
  for (var i = li; i < u.n; i += 256u) { s += partial[i]; }
  red[li] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  if (li == 0u) { normSq[0] = red[0]; }
}`

/**
 * AdamW with global-norm clipping folded in: g = grad * min(1, maxNorm / ||grad||).
 * Weight decay applies only where the decay bitmask is set (matrices and embeddings).
 */
const adamw = /* wgsl */ `
struct U { n: u32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32, maxNorm: f32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> W: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> M: array<f32>;
@group(0) @binding(4) var<storage, read_write> Vv: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<u32>;
@group(0) @binding(6) var<storage, read> normSq: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.n) { return; }
  let norm = sqrt(normSq[0]);
  var k = 1.0;
  if (norm > u.maxNorm && norm > 0.0) { k = u.maxNorm / norm; }
  let g = G[i] * k;
  let m = u.b1 * M[i] + (1.0 - u.b1) * g;
  let v = u.b2 * Vv[i] + (1.0 - u.b2) * g * g;
  M[i] = m;
  Vv[i] = v;
  var decay = 0.0;
  if (((mask[i >> 5u] >> (i & 31u)) & 1u) != 0u) { decay = u.lr * u.wd; }
  let w = W[i];
  W[i] = w - (u.lr * (m / u.bc1 / (sqrt(v / u.bc2) + u.eps)) + decay * w);
}`

/** Token-embedding gradient as a gather (no float atomics in WGSL): one thread per (token, channel). */
const embedBackTok = /* wgsl */ `
struct U { N: u32, d: u32, V: u32, tokOff: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dres: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.V * u.d) { return; }
  let v = i / u.d;
  let c = i % u.d;
  var s = 0.0;
  for (var n = 0u; n < u.N; n++) {
    if (idx[n] == v) { s += dres[n * u.d + c]; }
  }
  G[u.tokOff + i] += s;
}`

/** Position-embedding gradient: G[pos + t*d + c] += sum_b dres[b*T + t, c]. */
const embedBackPos = /* wgsl */ `
struct U { B: u32, T: u32, d: u32, posOff: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dres: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(256)
fn main(${FLAT_ARGS}) {
  ${FLAT_INDEX}
  if (i >= u.T * u.d) { return; }
  let t = i / u.d;
  let c = i % u.d;
  var s = 0.0;
  for (var b = 0u; b < u.B; b++) { s += dres[(b * u.T + t) * u.d + c]; }
  G[u.posOff + i] += s;
}`

export const KERNELS = {
  embed,
  layernorm,
  layernormBackInput,
  layernormBackParams,
  colsum,
  mmNN: matmul(false, false),
  mmNT: matmul(false, true),
  mmTN: matmul(true, false),
  add,
  geluFwd,
  geluBack,
  attnFwd,
  attnBack1,
  attnBack2,
  xent,
  meanReduce,
  normPartial,
  normFinal,
  adamw,
  embedBackTok,
  embedBackPos,
} as const

export type KernelName = keyof typeof KERNELS
