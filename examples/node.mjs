// npm install percent-nlu   →   node examples/node.mjs
import { NLU } from "percent-nlu";

const nlu = await NLU.load();
for (const [text, lang] of [
  ["how much is an 18% tip on $64?", "en"],
  ["quanto é 30% de desconto em 250?", "pt"],
  ["el alquiler subió de 1.450 a 1.595", "es"],
  ["20% of my salary", "en"],
]) {
  const r = nlu.predict(text, lang);
  console.log(`${text}\n  → ${r.status} ${r.tool ?? ""} ${JSON.stringify(r.args)} ${r.result ?? ""} (${r.timing.total_ms.toFixed(2)} ms)`);
}
