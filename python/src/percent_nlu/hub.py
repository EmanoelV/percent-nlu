"""Formato de distribuição (Hugging Face): config.json + model.safetensors, sem dependências além do torch.

safetensors = u64 little-endian (tamanho do cabeçalho) + cabeçalho JSON + bytes dos tensores (row-major).
uso: python -m percent_nlu.hub runs/r6/B2_ml/model.pt release/percent-nlu
"""
import json
import os
import struct
import sys

import torch

from percent_nlu.featurize import MAX_LEN, MAX_NG, N_FEATS
from percent_nlu.normalizer import LOCALES, lexicon
from percent_nlu.schema import INTENTS_A, INTENTS_B, MAX_SLOTS, ROLES, TOOLS

DTYPES = {torch.float32: "F32", torch.int64: "I64", torch.int8: "I8"}
BACK = {v: k for k, v in DTYPES.items()}


def save_safetensors(tensors, path, metadata=None):
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        t = t.detach().cpu().contiguous()
        b = t.view(torch.uint8).flatten().tolist() if t.numel() else []
        data = bytes(b)
        header[name] = {"dtype": DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def load_safetensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        buf = bytearray(raw[base + a:base + b])
        t = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
        out[name] = t.view(BACK[info["dtype"]]).reshape(info["shape"]).clone()
    return out


def save_pretrained(ckpt_path, out_dir, extra=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    os.makedirs(out_dir, exist_ok=True)
    save_safetensors({k: v.float() if v.is_floating_point() else v for k, v in ck["state"].items()},
                     os.path.join(out_dir, "model.safetensors"), {"format": "pt"})
    variant = a.get("variant", "A")
    cfg = {
        "model_type": "percent-nlu", "version": 1,
        "architecture": {"d_model": a["d"], "layers": a["layers"], "heads": a["heads"], "ff": a["ff"], "norm_first": True,
                         "activation": "gelu", "dropout": 0.1, "pooling": "mean", "use_slot_features": not a.get("no_feats", False)},
        "featurizer": {"buckets": a["buckets"], "max_len": MAX_LEN, "max_ngrams": MAX_NG, "ngram_range": [3, 5],
                       "hash": "fnv1a_32", "n_feats": N_FEATS, "max_slots": MAX_SLOTS},
        "vocab": ck["vocab"],
        "intents": INTENTS_B if variant == "B" else INTENTS_A, "roles": ROLES, "tools": TOOLS, "variant": variant,
        "calibration": {"temperature_intent": ck["temps"][0], "temperature_role": ck["temps"][1], "tau": ck["tau"]},
        "languages": list(LOCALES), "lexicons": {lg: lexicon(lg).d for lg in LOCALES},
        "training": {k: a[k] for k in ("epochs", "bs", "lr", "seed", "ls", "word_drop", "ema", "tau_target") if k in a},
        "n_parameters": sum(v.numel() for v in ck["state"].values()),
    }
    if extra:
        cfg.update(extra)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), ensure_ascii=False, indent=1)
    return cfg


def load_pretrained(model_dir):
    """Devolve um dicionário no formato do checkpoint de treino (para infer.Predictor)."""
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    arch = cfg["architecture"]
    state = load_safetensors(os.path.join(model_dir, "model.safetensors"))
    args = {"name": cfg.get("name", os.path.basename(os.path.abspath(model_dir))), "d": arch["d_model"],
            "layers": arch["layers"], "heads": arch["heads"], "ff": arch["ff"], "buckets": cfg["featurizer"]["buckets"],
            "variant": cfg["variant"], "no_feats": not arch["use_slot_features"]}
    cal = cfg["calibration"]
    return {"state": state, "vocab": cfg["vocab"], "args": args, "temps": [cal["temperature_intent"], cal["temperature_role"]],
            "tau": cal["tau"]}


if __name__ == "__main__":
    cfg = save_pretrained(sys.argv[1], sys.argv[2])
    print(f"salvo em {sys.argv[2]}: {cfg['n_parameters']:,} parâmetros, idiomas {cfg['languages']}")
