"""Normalizador multilíngue: motor único + léxico por idioma (percent_nlu/locales/<pasta>/numbers.json).

Extrai números (dígitos com separadores do idioma, %, moeda, multiplicadores, números por extenso) e mascara
como [N0]..[N3]. O mesmo JSON é usado pelo porte em JS (site/nlu.js) — mudou aqui, mude lá e rode a paridade.

Saída de normalize(text, lang):
  {"masked", "slots": [{"slot","value","surface","span","has_percent","has_currency"}], "overflow", "norm_text", "lang"}
"""
import json
import os
import re
import unicodedata

MAX_SLOTS = 4
LOCALES = {"pt": "pt_BR", "en": "en", "es": "es"}  # código → pasta em percent_nlu/locales/
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")

# formatos de número com dígitos: regex + regra de leitura
NUM_RE = {
    # 1.234,56 · 1234,5 · 3.5 (ponto decimal só se não for grupo de 3)
    "comma": r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?(?!\d)|\d+\.\d+|\d+(?:,\d+)?)",
    # 1,234.56 · 1234.5 · 1,5 (vírgula decimal só se não for grupo de 3)
    "dot": r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+,\d+|\d+(?:\.\d+)?)",
    # espanhol (Espanha × México): grupo de exatamente 3 dígitos = milhar, com ponto ou vírgula; senão decimal
    "auto3": r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?(?!\d)|\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+[.,]\d+|\d+)",
}


def unaccent(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def parse_digits(num, style):
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", num) and style in ("comma", "auto3"):
        return float(num.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", num) and style in ("dot", "auto3"):
        return float(num.replace(",", ""))
    return float(num.replace(",", "."))


class Lexicon:
    def __init__(self, lang):
        self.lang = lang
        d = json.load(open(os.path.join(_DIR, LOCALES[lang], "numbers.json"), encoding="utf-8"))
        self.d = d
        self.style = d["decimal"]
        self.mult_values = d["mult_values"]
        self.values = d["values"]
        self.hundred = d["hundred"]
        self.scales = d["scales"]
        self.fractions = d["fractions"]
        self.lead_one = set(d["lead_one"])
        self.connectors = set(d["connectors"])
        self.guard = set(d["connector_guard"])
        self.implicit = d["implicit_join"]
        self.hyphen = d["hyphen_join"]
        self.article_only = set(d["article_only"])
        self.protect = d["protect"]
        self.numword = {**self.values, **self.hundred, **self.scales, **self.fractions}
        pct, cur_pre, cur_pos, mult = d["pct"], d["cur_pre"], d["cur_pos"], d["mult"]
        self.pct_re = re.compile(pct)
        self.digit_re = re.compile(
            rf"(?P<cur>{cur_pre})?(?<![\w.,/])(?P<num>{NUM_RE[self.style]})(?P<mult>{mult})?(?P<pct>\s*{pct})?"
            rf"(?P<cur2>{cur_pos})?(?![\w/]|[.,]\d)")
        self.tail_re = re.compile(rf"(?P<pct>\s*{pct})?(?P<cur2>{cur_pos})?")
        self.curpre_re = re.compile(cur_pre + "$")
        self.stopwords = set(d.get("stopwords", []))


_LEX = {}


def lexicon(lang):
    if lang not in _LEX:
        _LEX[lang] = Lexicon(lang)
    return _LEX[lang]


def parse_words(words, L):
    total, cur = 0.0, 0.0
    for t in words:
        if t in L.connectors:
            continue
        if t in L.fractions:
            cur += L.fractions[t]
        elif t in L.lead_one:
            cur += 1
        elif t in L.values:
            cur += L.values[t]
        elif t in L.hundred:
            cur = (cur or 1) * L.hundred[t]
        elif t in L.scales:
            cur = (cur or 1) * L.scales[t]
            total += cur
            cur = 0
    return total + cur


def _place(L, w):
    """Maior valor que pode vir colado depois de w sem conector (en/es: 'twenty one', 'ciento veinte', 'mil doscientos')."""
    if w in L.scales:
        return L.scales[w]
    if w in L.hundred:
        return 100
    v = L.values.get(w, 0)
    if v >= 100 and v % 100 == 0:
        return 100
    if 20 <= v < 100 and v % 10 == 0:
        return 10
    return 0


def scan_words(t2, L, guard_pos=frozenset()):
    """Varre números por extenso. Regras: escala depois de número; conector + valor menor (pt 'vinte e um');
    junção implícita por casa decimal (en/es); 'hundred' multiplica (en); 'a hundred' (en)."""
    toks = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[a-z_]+", t2)]
    out, i = [], 0
    while i < len(toks):
        s0, e0, w = toks[i]
        if w in L.lead_one and w not in L.numword:
            nxt = toks[i + 1][2] if i + 1 < len(toks) else None
            if not (nxt and (nxt in L.hundred or nxt in L.scales) and not t2[e0:toks[i + 1][0]].strip()):
                i += 1; continue
        elif w not in L.numword:
            i += 1; continue
        seq, j, end = [w], i + 1, e0
        last = 1.0 if w in L.lead_one and w not in L.numword else L.numword[w]
        while j < len(toks):
            w2 = toks[j][2]
            gap = t2[toks[j - 1][1]:toks[j][0]]
            if gap.strip() and not (L.hyphen and gap == "-"):
                break
            if w2 in L.scales and seq[-1] not in L.scales:
                seq.append(w2); last = L.scales[w2]; end = toks[j][1]; j += 1; continue
            if w2 in L.hundred and seq[-1] not in L.hundred and seq[-1] not in L.scales and last < 100:
                seq.append(w2); last = L.hundred[w2]; end = toks[j][1]; j += 1; continue
            if w2 in L.connectors and toks[j][0] not in guard_pos and j + 1 < len(toks) \
                    and toks[j + 1][2] in L.numword and not t2[toks[j][1]:toks[j + 1][0]].strip():
                nv = L.numword[toks[j + 1][2]]
                if last >= 20 and nv < last and toks[j + 1][2] not in L.scales:
                    seq.append(toks[j + 1][2]); last = nv; end = toks[j + 1][1]; j += 2; continue
            if L.implicit and w2 in L.values and L.values[w2] < _place(L, seq[-1]):
                seq.append(w2); last = L.values[w2]; end = toks[j][1]; j += 1; continue
            break
        tail = L.tail_re.match(t2, end)
        pct, cur2 = bool(tail["pct"]), bool(tail["cur2"])
        e_final = tail.end() if (pct or cur2) else end
        pre = L.curpre_re.search(t2[:s0])
        cur = bool(pre) or cur2
        s_final = pre.start() if pre else s0
        alone_article = all(x in L.article_only for x in seq) and not pct
        if not alone_article:
            out.append((s_final, e_final, parse_words(seq, L), pct, cur))
        i = j
    return out


def normalize(text, lang="pt"):
    L = lexicon(lang)
    t_acc = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text.lower())).strip()
    t = unaccent(t_acc)
    guard_pos = frozenset(i for i, c in enumerate(t_acc) if c in L.guard) if len(t) == len(t_acc) else frozenset()
    found = []
    for m in L.digit_re.finditer(t):
        v = parse_digits(m["num"], L.style)
        if m["mult"]:
            v *= L.mult_values[m["mult"].strip()]
        found.append((m.start(), m.end(), v, bool(m["pct"]), bool(m["cur"] or m["cur2"])))
    t2 = t
    for a, b in L.protect:  # mesmo comprimento; "cento" de "por cento" não é número
        t2 = t2.replace(a, b)
    for s_, e_, v, pct, cur in scan_words(t2, L, guard_pos):
        if any(not (e_ <= s or s_ >= e) for s, e, *_ in found):
            continue
        found.append((s_, e_, v, pct, cur))
    found.sort()
    overflow = len(found) > MAX_SLOTS
    found = found[:MAX_SLOTS]
    out, slots, pos = [], [], 0
    for k, (s, e, v, pct, cur) in enumerate(found):
        out.append(t[pos:s])
        out.append((" $ " if cur else " ") + "[N" + "abcd"[k] + "]" + (" % " if pct else " "))
        slots.append({"slot": f"N{k}", "value": v, "surface": t[s:e].strip(), "span": [s, e],
                      "has_percent": pct, "has_currency": cur})
        pos = e
    out.append(t[pos:])
    masked = "".join(out)
    masked = re.sub(r"\d", "0", masked)  # dígitos restantes (10mb, 1/3) não vazam valores
    masked = re.sub(r"\[N([abcd])\]", lambda m: "[N%d]" % "abcd".index(m.group(1)), masked)
    masked = L.pct_re.sub(" % ", masked)  # "quantos por cento" -> "quantos %"
    masked = re.sub(r"\s+", " ", masked).strip()
    return {"masked": masked, "slots": slots, "overflow": overflow, "norm_text": t, "lang": lang}


def detect_lang(text, langs=("en", "pt", "es")):
    """Idioma pela contagem de palavras de função + palavras de número de cada léxico (empate → ordem de `langs`)."""
    toks = re.findall(r"[a-z]+", unaccent(text.lower()))
    best, best_sc = langs[0], -1
    for lg in langs:
        L = lexicon(lg)
        sw = {unaccent(w) for w in L.stopwords} | set(L.numword) | L.connectors
        sc = sum(t in sw for t in toks)
        if sc > best_sc:
            best, best_sc = lg, sc
    return best
