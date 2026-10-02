// Paridade Python × JS: compara normalização, tokens, probabilidades e decisão final.
// uso: node site/parity_test.mjs   (depois de python -m percent_nlu.export_web ...)
import { readFileSync } from "node:fs";
import { NLU, detectLang } from "../percent-nlu.js";

const dir = new URL("../model/", import.meta.url);
const M = JSON.parse(readFileSync(new URL("manifest.json", dir)));
const b = readFileSync(new URL("weights.bin", dir));
const nlu = new NLU(M, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
const cases = JSON.parse(readFileSync(new URL("./parity.json", import.meta.url)));

const fails = { detect: [], masked: [], values: [], ids: [], probs: [], decision: [] };
let maxProbDiff = 0;
const t0 = performance.now();
for (const c of cases) {
  const det = detectLang(c.text, nlu.lexicons, nlu.langs);
  if (det !== c.detected) fails.detect.push([c.text, c.detected, det]);
  const out = nlu.predict(c.text, c.lang);
  if (out.masked !== c.masked) { fails.masked.push([c.text, c.masked, out.masked]); continue; }
  const vals = out.numbers.map((n) => n.value);
  if (vals.length !== c.values.length || vals.some((v, i) => Math.abs(v - c.values[i]) > 1e-9 * Math.max(1, Math.abs(v)))) {
    fails.values.push([c.text, c.values, vals]); continue;
  }
  const ids = nlu.members[0].encode(out.masked, out.numbers.map((n, i) => ({ ...n, has_currency: c.has_currency[i] }))).ids;
  if (ids.join() !== c.ids.join()) { fails.ids.push([c.text, c.ids, ids]); continue; }
  if (!out._probs) continue;
  const d = Math.max(...out._probs.ip.map((p, i) => Math.abs(p - c.intent_probs[i])));
  maxProbDiff = Math.max(maxProbDiff, d);
  if (d > 1e-4) fails.probs.push([c.text, d]);
  const same = out.status === c.status && out.intent === c.intent &&
    JSON.stringify(Object.entries(out.roles).sort()) === JSON.stringify(Object.entries(c.roles).sort());
  if (!same) fails.decision.push([c.text, [c.status, c.intent, c.roles, c.confidence], [out.status, out.intent, out.roles, out.confidence]]);
}
const ms = performance.now() - t0;
console.log(`casos: ${cases.length}  tempo médio JS (node): ${(ms / cases.length).toFixed(3)} ms/pergunta`);
console.log(`Δprob de intenção máx: ${maxProbDiff.toExponential(2)}`);
for (const [k, v] of Object.entries(fails)) {
  console.log(`${k}: ${v.length} falhas`);
  for (const f of v.slice(0, 5)) console.log("   ", JSON.stringify(f));
}
process.exit(Object.values(fails).some((v) => v.length) ? 1 : 0);
