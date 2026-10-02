# Changelog

## 1.0.0 — 2026-10-02

First public release.

- One model for English, Portuguese (Brazil) and Spanish: 1,160,462 parameters, 1.2 MB int8.
- Six tools: `percent_of`, `what_percent`, `percent_increase`, `percent_decrease`, `apply_increase`, `apply_decrease`,
  plus `incomplete` and `out_of_scope` detection and a calibrated confidence gate.
- Dependency-free JavaScript runtime (browser, Node, Deno, Bun) with TypeScript types.
- Python package (PyTorch) with the model bundled.
