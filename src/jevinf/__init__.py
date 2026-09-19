"""jevinf -- a dedicated inference engine for NanoJev plus a Jev-compatible service layer.

Design discipline:
  * Do not copy any upstream source; upstream code is loaded by path via importlib (see upstream.py).
  * Segmented token ids are always reverse-derived from the upstream-produced leaf_tokens and cross-validated (see plan.py).
  * The decision-head math mirrors the latter half of the upstream DecisionModel.forward (see head.py), and is validated against upstream side by side.
  * The service-layer contract stays exactly as it is: reuse upstream validators, engine knobs go only through HTTP headers (see server.py).
"""

__version__ = "0.1.0"
