"""percent-nlu: tiny on-device model for natural-language percentage questions (English, Portuguese, Spanish).

    import percent_nlu
    nlu = percent_nlu.load()
    nlu("how much is an 18% tip on $64?", lang="en")   # {'status': 'call', 'tool': 'percent_of', 'result': 11.52, ...}
"""
import os

__version__ = "1.0.0"
_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def load(path=None):
    """Returns a predictor: predictor(text, lang=None) -> dict. lang: "en" | "pt" | "es" (None = detect)."""
    from percent_nlu.infer import Predictor
    return Predictor(path or _MODEL)
