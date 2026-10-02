// percent-nlu 1.0.0 — https://github.com/EmanoelV/percent-nlu — MIT License (c) 2026 Emanoel S. Vieira
// percent-nlu no navegador: normalizador multilíngue + featurização + Transformer (int8→f32) + decodificação restrita + tools.
// Porte 1:1 de percent_nlu/{normalizer,featurize,model,train.decode,infer,tools}.py. Sem dependências.
// Paridade com o Python: node site/parity_test.mjs

// ---------------- normalizador (motor único + léxico por idioma) ----------------
// Porte de percent_nlu/normalizer.py. Os léxicos (locales/*/numbers.json) vêm dentro do manifest.json do modelo.
const W = "[\\p{L}\\p{N}_]"; // \w do Python (Unicode)
const NB = `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`; // \b exato do Python (inclusive depois de "$")
const NUM_RE = {
  comma: "(?:\\d{1,3}(?:\\.\\d{3})+(?:,\\d+)?(?!\\d)|\\d+\\.\\d+|\\d+(?:,\\d+)?)",
  dot: "(?:\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?(?!\\d)|\\d+,\\d+|\\d+(?:\\.\\d+)?)",
  auto3: "(?:\\d{1,3}(?:\\.\\d{3})+(?:,\\d+)?(?!\\d)|\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?(?!\\d)|\\d+[.,]\\d+|\\d+)",
};
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const py = (re) => re.replace(/\\b/g, NB); // regex do léxico (sintaxe Python) → JS

export function unaccent(s) { return s.normalize("NFD").replace(/\p{Mn}/gu, ""); }

function parseDigits(num, style) {
  if (/^\d{1,3}(\.\d{3})+(,\d+)?$/.test(num) && (style === "comma" || style === "auto3"))
    return parseFloat(num.replace(/\./g, "").replace(",", "."));
  if (/^\d{1,3}(,\d{3})+(\.\d+)?$/.test(num) && (style === "dot" || style === "auto3"))
    return parseFloat(num.replace(/,/g, ""));
  return parseFloat(num.replace(",", "."));
}

export class Lexicon {
  constructor(d) {
    this.lang = d.lang; this.style = d.decimal; this.multValues = d.mult_values;
    this.values = d.values; this.hundred = d.hundred; this.scales = d.scales; this.fractions = d.fractions;
    this.leadOne = new Set(d.lead_one); this.connectors = new Set(d.connectors); this.guard = new Set(d.connector_guard);
    this.implicit = d.implicit_join; this.hyphen = d.hyphen_join; this.articleOnly = new Set(d.article_only);
    this.protect = d.protect; this.stopwords = new Set(d.stopwords.map(unaccent));
    this.numword = { ...this.values, ...this.hundred, ...this.scales, ...this.fractions };
    const pct = py(d.pct), curPre = py(d.cur_pre), curPos = py(d.cur_pos), mult = py(d.mult);
    this.pctG = new RegExp(pct, "gu");
    this.digitRe = new RegExp(`(?<cur>${curPre})?(?<![\\p{L}\\p{N}_.,/])(?<num>${NUM_RE[this.style]})(?<mult>${mult})?` +
      `(?<pct>\\s*${pct})?(?<cur2>${curPos})?(?![\\p{L}\\p{N}_/]|[.,]\\d)`, "gu");
    this.tailRe = new RegExp(`(?<pct>\\s*${pct})?(?<cur2>${curPos})?`, "uy");
    this.curpreRe = new RegExp(`${curPre}$`, "u");
  }
}

function parseWords(words, L) {
  let total = 0, cur = 0;
  for (const t of words) {
    if (L.connectors.has(t)) continue;
    if (has(L.fractions, t)) cur += L.fractions[t];
    else if (L.leadOne.has(t)) cur += 1;
    else if (has(L.values, t)) cur += L.values[t];
    else if (has(L.hundred, t)) cur = (cur || 1) * L.hundred[t];
    else if (has(L.scales, t)) { cur = (cur || 1) * L.scales[t]; total += cur; cur = 0; }
  }
  return total + cur;
}

function place(L, w) {
  if (has(L.scales, w)) return L.scales[w];
  if (has(L.hundred, w)) return 100;
  const v = has(L.values, w) ? L.values[w] : 0;
  if (v >= 100 && v % 100 === 0) return 100;
  if (v >= 20 && v < 100 && v % 10 === 0) return 10;
  return 0;
}

function scanWords(t2, L, guardPos) {
  const toks = [...t2.matchAll(/[a-z_]+/g)].map((m) => [m.index, m.index + m[0].length, m[0]]);
  const out = [];
  let i = 0;
  while (i < toks.length) {
    const [s0, e0, w] = toks[i];
    const lead = L.leadOne.has(w) && !has(L.numword, w);
    if (lead) {
      const nx = i + 1 < toks.length ? toks[i + 1] : null;
      if (!(nx && (has(L.hundred, nx[2]) || has(L.scales, nx[2])) && !t2.slice(e0, nx[0]).trim())) { i++; continue; }
    } else if (!has(L.numword, w)) { i++; continue; }
    const seq = [w];
    let j = i + 1, end = e0, last = lead ? 1 : L.numword[w];
    while (j < toks.length) {
      const w2 = toks[j][2], prev = seq[seq.length - 1];
      const gap = t2.slice(toks[j - 1][1], toks[j][0]);
      if (gap.trim() && !(L.hyphen && gap === "-")) break;
      if (has(L.scales, w2) && !has(L.scales, prev)) { seq.push(w2); last = L.scales[w2]; end = toks[j][1]; j++; continue; }
      if (has(L.hundred, w2) && !has(L.hundred, prev) && !has(L.scales, prev) && last < 100) {
        seq.push(w2); last = L.hundred[w2]; end = toks[j][1]; j++; continue;
      }
      if (L.connectors.has(w2) && !guardPos.has(toks[j][0]) && j + 1 < toks.length && has(L.numword, toks[j + 1][2])
          && !t2.slice(toks[j][1], toks[j + 1][0]).trim()) {
        const nv = L.numword[toks[j + 1][2]];
        if (last >= 20 && nv < last && !has(L.scales, toks[j + 1][2])) {
          seq.push(toks[j + 1][2]); last = nv; end = toks[j + 1][1]; j += 2; continue;
        }
      }
      if (L.implicit && has(L.values, w2) && L.values[w2] < place(L, prev)) {
        seq.push(w2); last = L.values[w2]; end = toks[j][1]; j++; continue;
      }
      break;
    }
    L.tailRe.lastIndex = end;
    const tail = L.tailRe.exec(t2);
    const pct = !!(tail && tail.groups.pct), cur2 = !!(tail && tail.groups.cur2);
    const eFinal = pct || cur2 ? L.tailRe.lastIndex : end;
    const pre = L.curpreRe.exec(t2.slice(0, s0));
    const cur = !!pre || cur2;
    const sFinal = pre ? pre.index : s0;
    const aloneArticle = seq.every((x) => L.articleOnly.has(x)) && !pct;
    if (!aloneArticle) out.push([sFinal, eFinal, parseWords(seq, L), pct, cur]);
    i = j;
  }
  return out;
}

export function normalize(text, L, maxSlots = 4) {
  const tAcc = text.toLowerCase().normalize("NFC").replace(/\s+/g, " ").trim();
  const t = unaccent(tAcc);
  const guardPos = new Set();
  if (t.length === tAcc.length) for (let i = 0; i < tAcc.length; i++) if (L.guard.has(tAcc[i])) guardPos.add(i);
  let found = [];
  for (const m of t.matchAll(L.digitRe)) {
    const g = m.groups;
    let v = parseDigits(g.num, L.style);
    if (g.mult) v *= L.multValues[g.mult.trim()];
    found.push([m.index, m.index + m[0].length, v, !!g.pct, !!(g.cur || g.cur2)]);
  }
  let t2 = t;
  for (const [a, b] of L.protect) t2 = t2.split(a).join(b);
  for (const f of scanWords(t2, L, guardPos)) {
    if (found.some(([s, e]) => !(f[1] <= s || f[0] >= e))) continue;
    found.push(f);
  }
  found.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const overflow = found.length > maxSlots;
  found = found.slice(0, maxSlots);
  const out = [], slots = [];
  let pos = 0;
  found.forEach(([s, e, v, pct, cur], k) => {
    out.push(t.slice(pos, s));
    out.push((cur ? " $ " : " ") + "[N" + "abcd"[k] + "]" + (pct ? " % " : " "));
    slots.push({ slot: `N${k}`, value: v, surface: t.slice(s, e).trim(), span: [s, e], has_percent: pct, has_currency: cur });
    pos = e;
  });
  out.push(t.slice(pos));
  let masked = out.join("").replace(/\d/g, "0").replace(/\[N([abcd])\]/g, (_, c) => `[N${"abcd".indexOf(c)}]`);
  masked = masked.replace(L.pctG, " % ").replace(/\s+/g, " ").trim();
  return { masked, slots, overflow, norm_text: t, lang: L.lang };
}

export function detectLang(text, lexicons, langs) {
  const toks = unaccent(text.toLowerCase()).match(/[a-z]+/g) || [];
  let best = langs[0], bestSc = -1;
  for (const lg of langs) {
    const L = lexicons[lg];
    const sc = toks.filter((t) => L.stopwords.has(t) || has(L.numword, t) || L.connectors.has(t)).length;
    if (sc > bestSc) { best = lg; bestSc = sc; }
  }
  return best;
}

// ---------------- featurização ----------------
const TOKEN_RE = /\[N\d\]|[a-z0-9_]+|[%$?!.,:;/+\-><]/g;

function fnv1a(s) {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(s)) { h ^= b; h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

function charNgrams(word, buckets, maxNg, specials) {
  if (specials.has(word) || !/^[\p{L}\p{N}]+$/u.test(word)) return [];
  const w = `<${word}>`, grams = [];
  for (let n = 3; n <= 5; n++) for (let i = 0; i + n <= w.length; i++) grams.push(w.slice(i, i + n));
  return grams.slice(0, maxNg).map((g) => 1 + (fnv1a(g) % (buckets - 1)));
}

function slotFeatures(numbers) {
  const vals = numbers.map((n) => n.value), mx = Math.max(...vals), mn = Math.min(...vals);
  return numbers.map((n, k) => {
    const v = n.value;
    const mag = v <= 0 ? 0 : Math.min(7, Math.max(0, Math.floor(Math.log10(v)) + 2));
    return [+n.has_percent, +n.has_currency, +Number.isInteger(v), +(v <= 100), mag, k, +(v === mx), +(v === mn), numbers.length];
  });
}

// ---------------- modelo ----------------
const erf = (x) => { // Abramowitz-Stegun 7.1.26 (erro < 1.5e-7)
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
};
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function linear(x, n, inD, W, b, outD) { // x [n,inD] · W[outD,inD]^T + b — blocos de 2 tokens × 4 saídas
  const y = new Float32Array(n * outD);
  let i = 0;
  for (; i + 1 < n; i += 2) {
    const xa = i * inD, xb = xa + inD, ya = i * outD, yb = ya + outD;
    let o = 0;
    for (; o + 3 < outD; o += 4) {
      const w0 = o * inD, w1 = w0 + inD, w2 = w1 + inD, w3 = w2 + inD;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      for (let k = 0; k < inD; k++) {
        const u = x[xa + k], v = x[xb + k], c0 = W[w0 + k], c1 = W[w1 + k], c2 = W[w2 + k], c3 = W[w3 + k];
        a0 += u * c0; a1 += u * c1; a2 += u * c2; a3 += u * c3;
        b0 += v * c0; b1 += v * c1; b2 += v * c2; b3 += v * c3;
      }
      const e0 = b ? b[o] : 0, e1 = b ? b[o + 1] : 0, e2 = b ? b[o + 2] : 0, e3 = b ? b[o + 3] : 0;
      y[ya + o] = a0 + e0; y[ya + o + 1] = a1 + e1; y[ya + o + 2] = a2 + e2; y[ya + o + 3] = a3 + e3;
      y[yb + o] = b0 + e0; y[yb + o + 1] = b1 + e1; y[yb + o + 2] = b2 + e2; y[yb + o + 3] = b3 + e3;
    }
    for (; o < outD; o++) {
      const wo = o * inD;
      let sa = b ? b[o] : 0, sb = sa;
      for (let k = 0; k < inD; k++) { sa += x[xa + k] * W[wo + k]; sb += x[xb + k] * W[wo + k]; }
      y[ya + o] = sa; y[yb + o] = sb;
    }
  }
  for (; i < n; i++) {
    const xo = i * inD, yo = i * outD;
    for (let o = 0; o < outD; o++) {
      let s = b ? b[o] : 0;
      const wo = o * inD;
      for (let k = 0; k < inD; k++) s += x[xo + k] * W[wo + k];
      y[yo + o] = s;
    }
  }
  return y;
}

function layerNorm(x, n, d, g, b) {
  const y = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let m = 0; for (let k = 0; k < d; k++) m += x[o + k]; m /= d;
    let v = 0; for (let k = 0; k < d; k++) { const z = x[o + k] - m; v += z * z; } v /= d;
    const r = 1 / Math.sqrt(v + 1e-5);
    for (let k = 0; k < d; k++) y[o + k] = (x[o + k] - m) * r * g[k] + b[k];
  }
  return y;
}

function softmax(a, temp = 1) {
  const z = Array.from(a, (v) => v / temp), m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m)), s = e.reduce((p, c) => p + c, 0);
  return e.map((v) => v / s);
}

class Member {
  constructor(spec, buf, M) {
    Object.assign(this, { d: spec.d, layers: spec.layers, heads: spec.heads, ff: spec.ff, temps: spec.temps, name: spec.name });
    this.M = M;
    this.vocab = new Map(spec.vocab.map((w, i) => [w, i]));
    this.specials = new Set(spec.vocab.slice(0, 2 + M.max_slots));
    this.w = {};
    for (const [name, t] of Object.entries(spec.tensors)) {
      const numel = t.shape.reduce((a, b) => a * b, 1);
      if (t.dtype === "f32") this.w[name] = new Float32Array(buf, t.offset, numel).slice();
      else {
        const q = new Int8Array(buf, t.offset, numel), rows = t.shape[0], cols = numel / rows;
        const sc = new Float32Array(buf, t.scale_offset, rows), f = new Float32Array(numel);
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) f[r * cols + c] = q[r * cols + c] * sc[r];
        this.w[name] = f;
      }
    }
  }

  encode(masked, numbers) {
    const M = this.M;
    const toks = (masked.match(TOKEN_RE) || []).slice(0, M.max_len);
    const ids = toks.map((t) => this.vocab.get(t) ?? 1);
    const ngs = toks.map((t) => charNgrams(t, M.buckets, M.max_ng, this.specials));
    const slotPos = new Array(M.max_slots).fill(-1);
    toks.forEach((t, i) => { if (t.startsWith("[N")) { const k = +t[2]; if (slotPos[k] < 0) slotPos[k] = i; } });
    const feats = slotFeatures(numbers);
    const tokFeats = toks.map(() => new Array(M.n_feats.length).fill(0));
    const isSlot = toks.map(() => 0);
    slotPos.forEach((p, k) => { if (p >= 0 && k < feats.length) { tokFeats[p] = feats[k].map((f) => f + 1); isSlot[p] = 1; } });
    return { toks, ids, ngs, feats: tokFeats, isSlot, slotPos };
  }

  forward(e) { // retorna {ip: probs de intenção calibradas, rp: [MAX_SLOTS][R] probs de papel}
    const { d, heads, ff, w, M } = this, L = e.ids.length, dh = d / heads;
    let x = new Float32Array(L * d);
    const WE = w["word.weight"], PE = w["pos.weight"], SE = w["slot_flag.weight"], NE = w["ngram.weight"];
    const acc = new Float32Array(d);
    for (let i = 0; i < L; i++) {
      const o = i * d, wr = e.ids[i] * d, pr = i * d, sr = e.isSlot[i] * d;
      for (let k = 0; k < d; k++) x[o + k] = WE[wr + k] + PE[pr + k] + SE[sr + k];
      const g = e.ngs[i];
      if (g.length) { // média dos n-gramas (soma em f32 e depois divide, como no PyTorch)
        acc.fill(0);
        for (const id of g) { const r = id * d; for (let k = 0; k < d; k++) acc[k] += NE[r + k]; }
        for (let k = 0; k < d; k++) x[o + k] += acc[k] / g.length;
      }
      e.feats[i].forEach((f, j) => { const t = w[`feat.${j}.weight`], r = f * d; for (let k = 0; k < d; k++) x[o + k] += t[r + k]; });
    }
    for (let l = 0; l < this.layers; l++) {
      const p = `enc.layers.${l}.`;
      const h = layerNorm(x, L, d, w[p + "norm1.weight"], w[p + "norm1.bias"]);
      const qkv = linear(h, L, d, w[p + "self_attn.in_proj_weight"], w[p + "self_attn.in_proj_bias"], 3 * d);
      const att = new Float32Array(L * d), sc = 1 / Math.sqrt(dh), row = new Float32Array(L), d3 = 3 * d;
      for (let hd = 0; hd < heads; hd++) {
        const qo = hd * dh, ko = d + hd * dh, vo = 2 * d + hd * dh;
        for (let i = 0; i < L; i++) {
          let mx = -Infinity;
          for (let j = 0; j < L; j++) {
            let s = 0;
            const qi = i * d3 + qo, kj = j * d3 + ko;
            for (let k = 0; k < dh; k++) s += qkv[qi + k] * qkv[kj + k];
            row[j] = s * sc; if (row[j] > mx) mx = row[j];
          }
          let z = 0; for (let j = 0; j < L; j++) { row[j] = Math.exp(row[j] - mx); z += row[j]; }
          for (let j = 0; j < L; j++) {
            const a = row[j] / z, ai = i * d + hd * dh, vj = j * d3 + vo;
            for (let k = 0; k < dh; k++) att[ai + k] += a * qkv[vj + k];
          }
        }
      }
      const o = linear(att, L, d, w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"], d);
      for (let i = 0; i < x.length; i++) x[i] += o[i];
      const h2 = layerNorm(x, L, d, w[p + "norm2.weight"], w[p + "norm2.bias"]);
      const f1 = linear(h2, L, d, w[p + "linear1.weight"], w[p + "linear1.bias"], ff);
      for (let i = 0; i < f1.length; i++) f1[i] = gelu(f1[i]);
      const f2 = linear(f1, L, ff, w[p + "linear2.weight"], w[p + "linear2.bias"], d);
      for (let i = 0; i < x.length; i++) x[i] += f2[i];
    }
    const hN = layerNorm(x, L, d, w["norm.weight"], w["norm.bias"]);
    const pooled = new Float32Array(d);
    for (let i = 0; i < L; i++) for (let k = 0; k < d; k++) pooled[k] += hN[i * d + k] / L;
    const head = (v, pre, nOut) => {
      const a = linear(v, 1, d, w[pre + ".0.weight"], w[pre + ".0.bias"], d).map(gelu);
      return linear(a, 1, d, w[pre + ".3.weight"], w[pre + ".3.bias"], nOut);
    };
    const ip = softmax(head(pooled, "intent_head", M.intents.length), this.temps[0]);
    const rp = e.slotPos.map((p) => {
      const q = Math.max(0, p);
      return softmax(head(hN.subarray(q * d, (q + 1) * d), "role_head", M.roles.length), this.temps[1]);
    });
    return { ip, rp };
  }
}

// ---------------- decodificação restrita ----------------
function* combinations(arr, k, start = 0, acc = []) {
  if (acc.length === k) { yield acc.slice(); return; }
  for (let i = start; i < arr.length; i++) { acc.push(arr[i]); yield* combinations(arr, k, i + 1, acc); acc.pop(); }
}
function* permutations(arr, k, used = new Set(), acc = []) {
  if (acc.length === k) { yield acc.slice(); return; }
  for (const a of arr) if (!used.has(a)) { used.add(a); acc.push(a); yield* permutations(arr, k, used, acc); acc.pop(); used.delete(a); }
}

function decode(ilp, rlp, nSlots, M) {
  const order = [...ilp.keys()].sort((a, b) => ilp[b] - ilp[a]);
  const top = M.intents[order[0]];
  if (top === "out_of_scope") return { status: "out_of_scope", intent: top, roles: {}, missing: [], conf: Math.exp(ilp[order[0]]) };
  let best = null;
  for (const ii of order.slice(0, 2)) {
    const it = M.intents[ii];
    if (!M.tools[it]) continue;
    const req = M.tools[it], slots = [...Array(nSlots).keys()], k = Math.min(req.length, nSlots);
    for (const chosen of combinations(req, k)) {
      for (const perm of permutations(slots, k)) {
        let sc = ilp[ii];
        const assign = new Map(perm.map((s, i) => [s, chosen[i]]));
        for (const s of slots) sc += rlp[s][M.roles.indexOf(assign.get(s) ?? "none")];
        if (!best || sc > best[0]) best = [sc, it, assign, req.filter((r) => !chosen.includes(r))];
      }
    }
  }
  const [sc, it, assign, missing] = best;
  const roles = {};
  for (const [s, r] of assign) roles[`N${s}`] = r;
  return { status: missing.length ? "incomplete" : "call", intent: it, roles, missing, conf: Math.exp(sc) };
}

// ---------------- tools ----------------
const r10 = (x) => Math.round(x * 1e10) / 1e10;
export const TOOLS = {
  percent_of: ({ rate, base }) => ({ result: r10(rate * base / 100), formula: `${rate} × ${base} / 100`, warnings: [] }),
  what_percent: ({ part, whole }) => { if (whole === 0) throw new Error("whole_zero"); return { result: r10(part / whole * 100), formula: `${part} / ${whole} × 100`, warnings: [] }; },
  percent_increase: ({ initial, final }) => { if (initial === 0) throw new Error("initial_zero"); return { result: r10((final - initial) / initial * 100), formula: `(${final} − ${initial}) / ${initial} × 100`, warnings: final < initial ? ["direction_mismatch"] : [] }; },
  percent_decrease: ({ initial, final }) => { if (initial === 0) throw new Error("initial_zero"); return { result: r10((initial - final) / initial * 100), formula: `(${initial} − ${final}) / ${initial} × 100`, warnings: final > initial ? ["direction_mismatch"] : [] }; },
  apply_increase: ({ base, rate }) => ({ result: r10(base * (1 + rate / 100)), formula: `${base} × (1 + ${rate}/100)`, warnings: [] }),
  apply_decrease: ({ base, rate }) => ({ result: r10(base * (1 - rate / 100)), formula: `${base} × (1 − ${rate}/100)`, warnings: rate > 100 ? ["rate_over_100"] : [] }),
};

// ---------------- preditor (ensemble) ----------------
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class NLU {
  /**
   * Carrega manifest.json + weights.bin. base: URL ou caminho da pasta do modelo (padrão: ./model/ ao lado deste arquivo).
   * Navegador: via fetch. Node: lê do disco (caminho ou file:// URL) sem fetch.
   */
  static async load(base, fetchFn = globalThis.fetch) {
    const t0 = now();
    // Sem base: URLs literais por arquivo, que bundlers (Vite, webpack/Next.js, Rollup, Parcel) reconhecem e copiam como assets.
    let manUrl, binUrl;
    if (base === undefined) {
      manUrl = new URL("./model/manifest.json", import.meta.url);
      binUrl = new URL("./model/weights.bin", import.meta.url);
    } else {
      let root = String(base);
      if (!root.endsWith("/")) root += "/";
      manUrl = `${root}manifest.json`;
      binUrl = `${root}weights.bin`;
    }
    const isNode = typeof process !== "undefined" && !!process.versions?.node && typeof window === "undefined";
    let M, buf;
    if (isNode && !/^https?:/.test(String(manUrl))) {
      // webpackIgnore/@vite-ignore: este ramo só roda no Node; bundlers de navegador não devem tentar resolver node:*
      const fs = await import(/* webpackIgnore: true */ /* @vite-ignore */ "node:fs/promises");
      const { pathToFileURL } = await import(/* webpackIgnore: true */ /* @vite-ignore */ "node:url");
      const toUrl = (u) => (String(u).startsWith("file:") ? new URL(String(u)) : pathToFileURL(String(u)));
      M = JSON.parse(await fs.readFile(toUrl(manUrl), "utf8"));
      const b = await fs.readFile(toUrl(binUrl));
      buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    } else {
      [M, buf] = await Promise.all([
        fetchFn(String(manUrl)).then((r) => r.json()),
        fetchFn(String(binUrl)).then((r) => r.arrayBuffer()),
      ]);
    }
    const t1 = now();
    const nlu = new NLU(M, buf);
    nlu.loadTiming = { fetch_ms: t1 - t0, init_ms: now() - t1, bytes: buf.byteLength };
    return nlu;
  }

  constructor(M, buf) {
    this.M = M;
    this.name = M.name;
    this.tau = M.tau;
    this.members = M.members.map((s) => new Member(s, buf, M));
    this.lexicons = Object.fromEntries(Object.entries(M.lexicons).map(([k, d]) => [k, new Lexicon(d)]));
    this.langs = M.detect_order; // ordem de desempate da detecção
  }

  /** lang: "pt" | "en" | "es" (recomendado: o idioma da página); sem lang, detecta pelas palavras. */
  predict(text, lang) {
    const M = this.M, T = {};
    let t = now();
    const lg = lang && this.lexicons[lang] ? lang : detectLang(text, this.lexicons, this.langs);
    const n = normalize(text, this.lexicons[lg], M.max_slots);
    T.normalize_ms = now() - t; t = now();
    const encs = this.members.map((m) => m.encode(n.masked, n.slots));
    T.featurize_ms = now() - t; t = now();
    const out = { lang: lg, masked: n.masked, numbers: n.slots.map(({ slot, surface, value, has_percent }) => ({ slot, surface, value, has_percent })), tau: this.tau };
    if (!encs[0].ids.length) { // sem nenhum token: nada a interpretar
      Object.assign(out, { intent: "out_of_scope", roles: {}, missing: [], confidence: 1, top_intents: [], status: "out_of_scope", tool: null, args: {} });
      T.forward_ms = 0; T.decode_ms = 0; T.total_ms = T.normalize_ms + T.featurize_ms;
      out.timing = T;
      return out;
    }
    const ps = this.members.map((m, i) => m.forward(encs[i]));
    T.forward_ms = now() - t; t = now();
    const nI = M.intents.length, nS = Math.min(M.max_slots, n.slots.length);
    const ip = Array.from({ length: nI }, (_, i) => ps.reduce((s, p) => s + p.ip[i], 0) / ps.length);
    const rp = Array.from({ length: M.max_slots }, (_, s) => Array.from({ length: M.roles.length }, (_, r) => ps.reduce((a, p) => a + p.rp[s][r], 0) / ps.length));
    const p = decode(ip.map(Math.log), rp.map((r) => r.map(Math.log)), nS, M);
    Object.assign(out, {
      intent: p.intent, roles: p.roles, missing: p.missing, confidence: Math.round(p.conf * 1e4) / 1e4,
      top_intents: M.intents.map((it, i) => [it, Math.round(ip[i] * 1e4) / 1e4]).sort((a, b) => b[1] - a[1]).slice(0, 3),
    });
    let status = p.status, args = {};
    if (status === "call") {
      const vals = Object.fromEntries(n.slots.map((s) => [s.slot, s.value]));
      for (const [slot, role] of Object.entries(p.roles)) if (role !== "none") args[role] = vals[slot];
      try { Object.assign(out, TOOLS[p.intent](args)); } catch (err) { Object.assign(out, { result: null, formula: null, warnings: [err.message] }); }
      if (p.conf < this.tau) status = "low_confidence";
    }
    Object.assign(out, { status, tool: TOOLS[p.intent] ? p.intent : null, args, _probs: { ip } });
    T.decode_ms = now() - t;
    T.total_ms = T.normalize_ms + T.featurize_ms + T.forward_ms + T.decode_ms;
    out.timing = T;
    return out;
  }
}
