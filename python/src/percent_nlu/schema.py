"""Esquema universal (independente de idioma): intenções, papéis e argumentos exigidos."""

TOOLS = {
    "percent_of": ("rate", "base"),
    "what_percent": ("part", "whole"),
    "percent_increase": ("initial", "final"),
    "percent_decrease": ("initial", "final"),
    "apply_increase": ("base", "rate"),
    "apply_decrease": ("base", "rate"),
}
OOS = "out_of_scope"
INCOMPLETE = "incomplete"  # usado só na variante B do A/B (classe explícita)

INTENTS_A = list(TOOLS) + [OOS]
INTENTS_B = list(TOOLS) + [OOS, INCOMPLETE]

ROLES = ["none", "rate", "base", "part", "whole", "initial", "final"]
MAX_SLOTS = 4
