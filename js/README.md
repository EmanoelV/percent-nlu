# percent-nlu

**Tiny on-device model (1.2 MB) that understands percentage questions in English, Portuguese and Spanish and
returns the right calculator tool call — in the browser or Node, in ~1 ms, no server, no API key. MIT.**

```bash
npm install percent-nlu
```

```js
import { NLU } from "percent-nlu";

const nlu = await NLU.load();                                   // bundled model
const r = nlu.predict("how much is an 18% tip on $64?", "en");  // "en" | "pt" | "es" (omit to auto-detect)
// r.status === "call" → r.tool === "percent_of", r.args → { rate: 18, base: 64 }, r.result → 11.52
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

CDN (no bundler):

```js
import { NLU } from "https://cdn.jsdelivr.net/npm/percent-nlu@1/percent-nlu.js";
const nlu = await NLU.load("https://cdn.jsdelivr.net/npm/percent-nlu@1/model/");
```

`status`: `call` (execute) · `low_confidence` (confirm with the user) · `incomplete` (`missing` says which number to ask
for) · `out_of_scope`. Tools: `percent_of`, `what_percent`, `percent_increase`, `percent_decrease`, `apply_increase`,
`apply_decrease`. Types in `percent-nlu.d.ts`.

Docs, accuracy and the live demo: [github.com/EmanoelV/percent-nlu](https://github.com/EmanoelV/percent-nlu) ·
model on [Hugging Face](https://huggingface.co/EmanoelV/percent-nlu) · [demo](https://huggingface.co/spaces/EmanoelV/percent-nlu-demo).
