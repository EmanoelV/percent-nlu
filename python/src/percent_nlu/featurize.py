"""Tokenização (palavra + n-gramas de caractere com hash FNV-1a, portável para JS) e features de slot."""
import math
import re
from collections import Counter

from percent_nlu.schema import ROLES, MAX_SLOTS

TOKEN_RE = re.compile(r"\[N\d\]|[a-z0-9_]+|[%$?!.,:;/+\-><]")
SPECIALS = ["<pad>", "<unk>"] + [f"[N{k}]" for k in range(MAX_SLOTS)]
MAX_LEN = 64
MAX_NG = 24


def tokenize(masked):
    return TOKEN_RE.findall(masked)[:MAX_LEN]


def fnv1a(s):
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def char_ngrams(word, buckets, nmin=3, nmax=5):
    if word in SPECIALS or not word.isalnum():
        return []
    w = f"<{word}>"
    grams = [w[i:i + n] for n in range(nmin, nmax + 1) for i in range(len(w) - n + 1)]
    return [1 + fnv1a(g) % (buckets - 1) for g in grams][:MAX_NG]  # 0 = padding


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, rows, min_freq=3):
        c = Counter(t for r in rows for t in tokenize(r["text_masked"]))
        words = sorted(w for w, n in c.items() if n >= min_freq and w not in SPECIALS)
        return cls(SPECIALS + words)

    def __len__(self):
        return len(self.itos)

    def id(self, w):
        return self.stoi.get(w, 1)


N_FEATS = [2, 2, 2, 2, 8, MAX_SLOTS, 2, 2, MAX_SLOTS + 1]  # cardinalidade de cada feature


def slot_features(numbers):
    vals = [n["value"] for n in numbers]
    feats = []
    for k, n in enumerate(numbers):
        v = n["value"]
        mag = 0 if v <= 0 else min(7, max(0, int(math.floor(math.log10(v))) + 2))
        feats.append([int(n["has_percent"]), int(n["has_currency"]), int(float(v).is_integer()), int(v <= 100),
                      mag, k, int(v == max(vals)), int(v == min(vals)), len(numbers)])
    return feats


def encode(row, vocab, buckets, intents, use_feats=True):
    toks = tokenize(row["text_masked"])
    ids = [vocab.id(t) for t in toks]
    ngs = [char_ngrams(t, buckets) for t in toks]
    slot_pos = [-1] * MAX_SLOTS
    for i, t in enumerate(toks):
        if t.startswith("[N"):
            k = int(t[2])
            if slot_pos[k] < 0:
                slot_pos[k] = i
    feats = slot_features(row["numbers"]) if use_feats else []
    tok_feats = [[0] * len(N_FEATS) for _ in toks]
    tok_isslot = [0] * len(toks)
    for k, p in enumerate(slot_pos):
        if p >= 0 and k < len(feats):
            tok_feats[p] = [f + 1 for f in feats[k]] if use_feats else [0] * len(N_FEATS)  # 0 = sem feature
            tok_isslot[p] = 1
    role_y = [-100] * MAX_SLOTS
    for k in range(min(MAX_SLOTS, len(row["numbers"]))):
        if slot_pos[k] >= 0:
            role_y[k] = ROLES.index(row["roles"].get(f"N{k}", "none"))
    intent = row["intent"]
    return {"ids": ids, "ngs": ngs, "feats": tok_feats, "slot_pos": slot_pos, "role_y": role_y,
            "intent_y": intents.index(intent) if intent in intents else -100}
