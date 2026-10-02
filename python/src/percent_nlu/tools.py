"""As 6 tools de porcentagem (v1). Retornam {result, formula, warnings}."""


def _r(x):
    return round(x, 10)


def percent_of(rate, base):
    return {"result": _r(rate * base / 100), "formula": f"{rate} × {base} / 100", "warnings": []}


def what_percent(part, whole):
    if whole == 0:
        raise ValueError("whole_zero")
    return {"result": _r(part / whole * 100), "formula": f"{part} / {whole} × 100", "warnings": []}


def percent_increase(initial, final):
    if initial == 0:
        raise ValueError("initial_zero")
    w = ["direction_mismatch"] if final < initial else []
    return {"result": _r((final - initial) / initial * 100),
            "formula": f"({final} − {initial}) / {initial} × 100", "warnings": w}


def percent_decrease(initial, final):
    if initial == 0:
        raise ValueError("initial_zero")
    w = ["direction_mismatch"] if final > initial else []
    return {"result": _r((initial - final) / initial * 100),
            "formula": f"({initial} − {final}) / {initial} × 100", "warnings": w}


def apply_increase(base, rate):
    return {"result": _r(base * (1 + rate / 100)), "formula": f"{base} × (1 + {rate}/100)", "warnings": []}


def apply_decrease(base, rate):
    w = ["rate_over_100"] if rate > 100 else []
    return {"result": _r(base * (1 - rate / 100)), "formula": f"{base} × (1 − {rate}/100)", "warnings": w}


REGISTRY = {f.__name__: f for f in
            [percent_of, what_percent, percent_increase, percent_decrease, apply_increase, apply_decrease]}


def call(tool, args):
    return REGISTRY[tool](**args)
