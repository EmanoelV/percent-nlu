"""Inferência: texto -> {status, tool, args, result, ...} com o checkpoint treinado (ou um ensemble)."""
import json
import os

import torch
import torch.nn.functional as F

from percent_nlu import tools
from percent_nlu.featurize import Vocab, encode
from percent_nlu.model import PercentNLU, collate
from percent_nlu.normalizer import normalize, detect_lang
from percent_nlu.schema import INTENTS_A, INTENTS_B, MAX_SLOTS
from percent_nlu.decode import decode


class Predictor:
    """Predictor(caminho) aceita um checkpoint de treino (.pt) ou uma pasta de release (config.json + model.safetensors)."""

    def __init__(self, ckpt_path):
        if os.path.isdir(ckpt_path):
            from percent_nlu.hub import load_pretrained
            ck = load_pretrained(ckpt_path)
        else:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.args = a
        self.variant = a.get("variant", "A")
        self.intents = INTENTS_B if self.variant == "B" else INTENTS_A
        self.vocab = Vocab(ck["vocab"])
        self.buckets = a["buckets"]
        self.use_feats = not a.get("no_feats", False)
        self.temps = ck["temps"]
        self.tau = ck["tau"]
        self.model = PercentNLU(len(self.vocab), self.buckets, len(self.intents), a["d"], a["layers"], a["heads"],
                                a["ff"], use_feats=self.use_feats)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.name = a["name"]

    @torch.no_grad()
    def probs(self, n):
        """Probabilidades calibradas (intenção [I], papéis [MAX_SLOTS, R]) para um texto já normalizado."""
        row = {"text_masked": n["masked"], "numbers": n["slots"], "roles": {}, "intent": None}
        enc = encode(row, self.vocab, self.buckets, self.intents, self.use_feats)
        ids, ngs, feats, isslot, slot_pos, _, _ = collate([enc], torch.device("cpu"))
        il, rl = self.model(ids, ngs, feats, isslot, slot_pos)
        return F.softmax(il / self.temps[0], -1)[0], F.softmax(rl / self.temps[1], -1)[0]

    def __call__(self, text, lang=None):
        """lang: "pt" | "en" | "es" (recomendado); sem lang, detecta pelas palavras."""
        n = normalize(text, lang or detect_lang(text))
        ip, rp = self.probs(n)
        return finish(n, ip, rp, self.intents, self.variant, self.tau)


class EnsemblePredictor:
    """Média das probabilidades calibradas dos membros. Gate: conf >= tau e (se agree) todos os membros
    decodificam a mesma chamada. Manifesto: {"name", "members": [ckpt...], "tau", "agree"}."""

    def __init__(self, manifest_path):
        m = json.load(open(manifest_path))
        self.members = [Predictor(p) for p in m["members"]]
        self.name, self.tau, self.agree = m["name"], m["tau"], m["agree"]
        self.intents, self.variant = self.members[0].intents, self.members[0].variant

    def __call__(self, text, lang=None):
        n = normalize(text, lang or detect_lang(text))
        ps = [p.probs(n) for p in self.members]
        ip = torch.stack([a for a, _ in ps]).mean(0)
        rp = torch.stack([b for _, b in ps]).mean(0)
        k = min(MAX_SLOTS, len(n["slots"]))
        votes = [decode(a.log().tolist(), b.log().tolist(), k, self.intents, self.variant) for a, b in ps]
        agree = len({(v["status"], v["intent"], tuple(sorted(v["roles"].items()))) for v in votes}) == 1
        return finish(n, ip, rp, self.intents, self.variant, self.tau, force_low=self.agree and not agree,
                      extra={"members_agree": agree})


def finish(n, ip, rp, intents, variant, tau, force_low=False, extra=None):
    p = decode(ip.log().tolist(), rp.log().tolist(), min(MAX_SLOTS, len(n["slots"])), intents, variant)
    top_intents = sorted(zip(intents, ip.tolist()), key=lambda x: -x[1])[:3]
    out = {"lang": n.get("lang"), "masked": n["masked"], "numbers": [{k: s[k] for k in ("slot", "surface", "value", "has_percent")}
                                              for s in n["slots"]],
           "intent": p["intent"], "roles": p["roles"], "missing": p["missing"],
           "confidence": round(p["conf"], 4), "tau": tau,
           "top_intents": [[i, round(v, 4)] for i, v in top_intents]}
    out.update(extra or {})
    status = p["status"]
    args = {}
    if status == "call":
        vals = {s["slot"]: s["value"] for s in n["slots"]}
        args = {role: vals[slot] for slot, role in p["roles"].items() if role != "none"}
        try:
            res = tools.call(p["intent"], args)
            out.update(result=res["result"], formula=res["formula"], warnings=res["warnings"])
        except ValueError as e:
            out.update(result=None, formula=None, warnings=[str(e)])
        if p["conf"] < tau or force_low:
            status = "low_confidence"
    out["status"] = status
    out["tool"] = p["intent"] if p["intent"] in tools.REGISTRY else None
    out["args"] = args
    return out


def load_predictor(path):
    return EnsemblePredictor(path) if path.endswith(".json") else Predictor(path)
