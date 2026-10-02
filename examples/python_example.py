# pip install "percent-nlu @ git+https://github.com/EmanoelV/percent-nlu#subdirectory=python"
# python examples/python_example.py
import percent_nlu

nlu = percent_nlu.load()
for text, lang in [("how much is an 18% tip on $64?", "en"), ("quanto é 30% de desconto em 250?", "pt"),
                   ("el alquiler subió de 1.450 a 1.595", "es"), ("20% of my salary", "en")]:
    r = nlu(text, lang=lang)
    print(text, "→", r["status"], r["tool"], r["args"], r.get("result"))
