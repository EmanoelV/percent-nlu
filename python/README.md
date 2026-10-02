# percent-nlu

**Tiny model (1.16M parameters) that understands percentage questions in English, Portuguese and Spanish and
returns the right calculator tool call with every number in its role. Needs only PyTorch; the model is bundled. MIT.**

```bash
pip install "percent-nlu @ git+https://github.com/EmanoelV/percent-nlu#subdirectory=python"
```

```python
import percent_nlu

nlu = percent_nlu.load()
nlu("how much is an 18% tip on $64?", lang="en")
# {'status': 'call', 'tool': 'percent_of', 'args': {'rate': 18.0, 'base': 64.0}, 'result': 11.52, 'confidence': ..., ...}
nlu("quanto é 30% de desconto em 250?", lang="pt")["result"]   # 75.0
nlu("el alquiler subió de 1.450 a 1.595", lang="es")["tool"]    # 'percent_increase'
```

`status`: `call` (execute) · `low_confidence` (confirm) · `incomplete` (`missing` lists the number to ask for) ·
`out_of_scope`. The same model runs in the browser via the npm package `percent-nlu`.

Docs, accuracy and demo: [github.com/EmanoelV/percent-nlu](https://github.com/EmanoelV/percent-nlu) ·
[Hugging Face](https://huggingface.co/EmanoelV/percent-nlu) · [live demo](https://huggingface.co/spaces/EmanoelV/percent-nlu-demo).
