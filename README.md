# percent-nlu — natural-language percentage calculator model (English, Portuguese, Spanish)

**percent-nlu is a tiny open-source language model (1.16M parameters, 1.2 MB) that understands
percentage questions written in plain English, Portuguese or Spanish and turns them into the right calculation —
entirely in the browser, in about a millisecond, with no server and no API key.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![npm](https://img.shields.io/npm/v/percent-nlu)](https://www.npmjs.com/package/percent-nlu)
[![PyPI](https://img.shields.io/pypi/v/percent-nlu)](https://pypi.org/project/percent-nlu/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/EmanoelV/percent-nlu)
[![Demo](https://img.shields.io/badge/demo-live-blue)](https://huggingface.co/spaces/EmanoelV/percent-nlu-demo)

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/EmanoelV/percent-nlu-demo) · [GitHub Pages](https://emanoelv.github.io/percent-nlu/) ·
**In production:** [icupu.com Percentage Calculator](https://icupu.com/percentage-calculator)

```text
"how much is an 18% tip on $64?"          → percent_of(rate=18, base=64)               = 11.52
"quanto é 30% de desconto em 250?"        → percent_of(rate=30, base=250)              = 75
"el alquiler subió de 1.450 a 1.595"      → percent_increase(initial=1450, final=1595) = 10 %
"I invested 2,000 and now I have 2,600"   → percent_increase(initial=2000, final=2600) = 30 %
"200 minus 15%"                           → apply_decrease(base=200, rate=15)          = 170
"20% of my salary"                        → incomplete: missing base
"what percentage of the earth is water"   → out_of_scope
```

## Why

Percentage questions are among the most searched calculations on the web, but people do not type them the way calculator
forms expect: *"I got 38 out of 50, what's my grade?"*, *"rent went up from 1,450 to 1,595"*, *"quanto fica 1.200 com 15% de
desconto?"*. Regex-based "natural language" inputs break on phrasing they have not seen, and calling an LLM API for every
keystroke is slow, costs money and sends user input to a third party.

percent-nlu sits in between:

| | regex / pattern rules | **percent-nlu** | LLM API |
|---|---|---|---|
| Handles new phrasings | ✗ | ✓ (trained on ~2,700 templates in 3 languages) | ✓ |
| Runs in the browser, offline | ✓ | ✓ (1.2 MB, no dependencies) | ✗ |
| Latency | < 1 ms | ~1–5 ms on CPU | 300–3000 ms |
| Cost per question | 0 | 0 | > 0 |
| Arithmetic always exact | ✓ | ✓ (the model never computes; plain code does) | ✗ (can hallucinate) |
| Knows when it is unsure | ✗ | ✓ (calibrated confidence, `incomplete`, `out_of_scope`) | partly |

## Install

### JavaScript (browser, Node, Deno, Bun)

```bash
npm install percent-nlu
```

```js
import { NLU } from "percent-nlu";

const nlu = await NLU.load();                       // loads the bundled model (1.2 MB)
const r = nlu.predict("how much is an 18% tip on $64?", "en");   // "en" | "pt" | "es"

if (r.status === "call") console.log(r.tool, r.args, r.result);    // percent_of {rate: 18, base: 64} 11.52
```

### Bundlers and frameworks (Vite, Next.js, webpack, Rollup)

`NLU.load()` with no argument references the model files with `new URL(..., import.meta.url)`, so Vite and webpack 5
(Next.js) copy `manifest.json` and `weights.bin` into your build automatically (tested with Vite 6 and webpack 5).
The model must load in the browser, not during server rendering — in Next.js:

```jsx
"use client";
import { useEffect, useState } from "react";

export function usePercentNLU() {
  const [nlu, setNlu] = useState(null);
  useEffect(() => { import("percent-nlu").then(({ NLU }) => NLU.load()).then(setNlu); }, []);
  return nlu;   // nlu?.predict("how much is an 18% tip on $64?", "en")
}
```

Prefer to serve the files yourself? Copy `node_modules/percent-nlu/model/` to your static folder (e.g. `public/percent-nlu/`)
and call `NLU.load("/percent-nlu/")`.

No build step? Use a CDN:

```html
<script type="module">
  import { NLU } from "https://cdn.jsdelivr.net/npm/percent-nlu@1/percent-nlu.js";
  const nlu = await NLU.load("https://cdn.jsdelivr.net/npm/percent-nlu@1/model/");
  console.log(nlu.predict("quanto é 15% de 80?", "pt").result);   // 12
</script>
```

### Python

```bash
pip install percent-nlu
```

```python
import percent_nlu

nlu = percent_nlu.load()                            # bundled model, needs only PyTorch
nlu("cuánto es el 15% de 80?", lang="es")
# {'status': 'call', 'tool': 'percent_of', 'args': {'rate': 15.0, 'base': 80.0}, 'result': 12.0, ...}
```

## What it understands

| tool | question type | English | Português | Español |
|---|---|---|---|---|
| `percent_of` | X% of Y | *what is 15% of 80* | *quanto é 15% de 80* | *cuánto es el 15% de 80* |
| `what_percent` | X is what % of Y | *38 out of 50 as a percentage* | *38 é quantos % de 50* | *38 qué porcentaje es de 50* |
| `percent_increase` | % change, up | *from 80 to 100 in percent* | *de 80 para 100 aumentou quanto %* | *subió de 80 a 100, qué porcentaje* |
| `percent_decrease` | % change, down | *was $80, now $60, what % off* | *custava 80 e paguei 60, desconto de quanto* | *costaba 80, ahora 60, qué descuento es* |
| `apply_increase` | Y plus X% | *1,400 rent going up 5%* | *aluguel de 1.400 com reajuste de 5%* | *alquiler de 1.400 que sube un 5%* |
| `apply_decrease` | Y minus X% | *80 dollars with 25% off* | *80 reais com 25% de desconto* | *80 euros con 25% de descuento* |

Numbers can be written as digits in each language's format (`1,234.56` / `1.234,56`), with currencies (`$`, `€`, `R$`), with
multipliers (`5k`, `2 mil`, `1.2 million`) or as words (*"one hundred and twenty-five"*, *"dois mil e trezentos"*,
*"doscientos cincuenta y tres mil"*). The `%` sign is optional (*"15 of 80"*).

### Output

Every prediction has a `status`:

- `call` — confident: execute `tool(args)` and show `result`.
- `low_confidence` — best guess below the confidence threshold: show it and ask the user to confirm.
- `incomplete` — the intent is clear but a number is missing (*"20% of my salary"*): `missing` lists what to ask for.
- `out_of_scope` — not a percentage calculation (battery levels, polls, unit conversions, plain arithmetic, chit-chat).

See [`js/percent-nlu.d.ts`](js/percent-nlu.d.ts) for the full type.

## How it works

1. **Normalizer** (deterministic rules + a small JSON lexicon per language) finds every number, its percent/currency
   markers and its value, and masks it: `how much is an 18% tip on $64?` → `how much is an [N0] % tip on $ [N1] ?`.
2. **Transformer encoder** (2 layers, d=128) reads the masked text using word embeddings + hashed character n-grams
   (robust to typos and missing accents) + small per-number features, and predicts the intent and a role for each `[Nk]`.
3. **Constrained decoding** assigns each required argument of the chosen tool to a different number.
4. **Calibrated confidence**: `status = call` only above a threshold tuned for ~98–99% precision.
5. **Plain code** computes the result — the model never does arithmetic.

One model serves all three languages; the language only selects the number lexicon (pass it when you know the page
language; otherwise it is detected).

## Accuracy

Exact = tool **and** every argument correct. Gold sets are hand-checked questions never used in training.

| set | n | exact | out-of-scope wrongly accepted | precision of `call` | share answered with `call` |
|---|---|---|---|---|---|
| gold v0 (pt-BR) | 242 | 97.6% | 6.0% | 98.7% | 95.7% |
| gold v1 (pt-BR, hard) | 112 | 90.3% | 0.0% | 96.1% | 81.7% |
| gold v1 (English) | 118 | 95.7% | 10.0% | 100.0% | 91.3% |
| gold v1 (Español) | 118 | 94.6% | 0.0% | 98.8% | 92.4% |
| synthetic test, pt-BR (unseen templates) | 3812 | 90.5% | 7.0% | 98.1% | 82.6% |
| synthetic test, English (unseen templates) | 3815 | 95.1% | 7.4% | 99.3% | 85.1% |
| synthetic test, Español (unseen templates) | 3812 | 98.2% | 19.4% | 96.4% | 91.2% |

Size 1,160,462 parameters · 1.2 MB int8 · JavaScript ~1.4 ms per question on an Apple M1, ~4.6 ms on a
mid-range Android phone · Python/PyTorch p50 0.6 ms.

## Limitations

- Six tools only, at most four numbers per question, no chained calculations (*"20% off and then 10% tax"*).
- Not supported yet: fractions (*1/4*), decimals in words (*"seven point five"*), ranges.
- In Spanish, `1,500` is read as 1500 (groups of exactly three digits are thousands for both `.` and `,`).
- Trained on synthetic questions; real users will find new phrasings — that is what the `status` field is for.

## FAQ

**How do I add natural-language input to a percentage calculator?**
Install `percent-nlu`, call `nlu.predict(text, lang)` and, when `status` is `call`, run the returned `tool` with `args`
(or just use `result`). It runs client-side, so there is no backend to maintain.

**Does it work offline / without a server?**
Yes. The model (1.2 MB) and the runtime are plain files; after the first load everything runs on the device.

**Which languages are supported?**
English, Portuguese (Brazil) and Spanish (Spain and Latin America), with each language's number formats and number words.

**How is this different from asking ChatGPT or another LLM?**
It answers in milliseconds, costs nothing per query, keeps the text on the user's device and never makes arithmetic
mistakes, because it only decides *which* calculation to run and with *which* numbers.

**Can I use it commercially?**
Yes — MIT license.

**Can I train it on my own data?**
The inference code and weights are open. The training data generator is not part of this release.

## Em português

**percent-nlu** é um modelo de linguagem minúsculo e de código aberto que entende perguntas de porcentagem escritas do jeito
que as pessoas falam — *"quanto é 30% de desconto em 250?"*, *"investi 100 e agora tenho 110"*, *"18 é quantos por cento de 72?"*
— e escolhe o cálculo certo. Roda no navegador, offline, em ~1 ms. Ideal para uma **calculadora de porcentagem com
linguagem natural**. Licença MIT.

## En español

**percent-nlu** es un modelo de lenguaje diminuto y de código abierto que entiende preguntas de porcentaje escritas en
lenguaje natural — *"cuánto es el 15% de 80?"*, *"el precio bajó de 49,99 a 39,99"*, *"súmale el 21% de iva a 120"* — y elige
el cálculo correcto. Funciona en el navegador, sin servidor, en ~1 ms. Ideal para una **calculadora de porcentajes en
lenguaje natural**. Licencia MIT.

## Citation

```bibtex
@software{percent_nlu,
  title  = {percent-nlu: a tiny on-device model for natural-language percentage questions},
  author = {Emanoel S. Vieira},
  year   = {2026},
  url    = {https://github.com/EmanoelV/percent-nlu},
  version = {1.0.0}
}
```

## License

[MIT](LICENSE) © 2026 Emanoel S. Vieira
