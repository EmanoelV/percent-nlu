"""Decodificação com restrição: escolhe intenção + atribuição de papéis exigidos pela tool (um slot por papel)."""
import itertools
import math

from percent_nlu.schema import TOOLS, OOS, INCOMPLETE, ROLES


def decode(intent_logp, role_logp, n_slots, intents, variant):
    """intent_logp: [I], role_logp: [MAX_SLOTS, R] (log-probs calibradas). Retorna dict de predição."""
    order = sorted(range(len(intents)), key=lambda i: -intent_logp[i])
    top = intents[order[0]]
    if top == OOS:
        return {"status": "out_of_scope", "intent": OOS, "roles": {}, "missing": [], "conf": math.exp(intent_logp[order[0]])}
    if top == INCOMPLETE:
        return {"status": "incomplete", "intent": INCOMPLETE, "roles": {}, "missing": [], "conf": math.exp(intent_logp[order[0]])}
    best = None
    none_i = ROLES.index("none")
    for ii in order[:2]:
        it = intents[ii]
        if it not in TOOLS:
            continue
        req = TOOLS[it]
        slots = list(range(n_slots))
        # atribuições: cada papel exigido -> slot distinto ou ausente (só se faltarem slots)
        k = min(len(req), n_slots)
        for chosen_roles in itertools.combinations(req, k):
            for perm in itertools.permutations(slots, k):
                sc = intent_logp[ii]
                assign = dict(zip(perm, chosen_roles))
                for s in slots:
                    r = assign.get(s, "none")
                    sc += role_logp[s][ROLES.index(r)]
                missing = [r for r in req if r not in chosen_roles]
                if best is None or sc > best[0]:
                    best = (sc, it, assign, missing)
    sc, it, assign, missing = best
    roles = {f"N{s}": r for s, r in assign.items()}
    status = "incomplete" if missing else "call"
    return {"status": status, "intent": it, "roles": roles, "missing": missing, "conf": math.exp(sc)}
