"""Conformance + boundary tests for the Jev-compatible API service layer.

The single most important one: **does the service layer forward the engine output verbatim**.
The approach is to take the offline `jevinf eval` golden and compare argmax against the HTTP
response question by question (should be 100%). That way "service layer == engine" is measured,
not assumed; stacked on the previously measured "engine == upstream oracle (100%)", it
transitively yields "service layer == oracle".

The rest is contract boundaries: strict validation, limits, knob headers, error codes.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from jevinf import upstream
from jevinf.bench import compare

import argparse

BASE = "http://127.0.0.1:8226"
GOLDEN = "data/golden32.json"


def post(body: bytes, headers: dict | None = None):
    req = urllib.request.Request(BASE + "/api/evaluate", data=body, method="POST",
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw[:200].decode("utf-8", "replace")}
        return err.code, dict(err.headers), parsed


def main() -> int:
    global BASE, GOLDEN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--split", required=True, help="dev split (jsonl)")
    args = ap.parse_args()
    BASE, GOLDEN = args.base, args.golden

    payload = upstream.read_split(args.split)
    req32 = {"states": payload["states"][:32]}
    body = json.dumps(req32).encode()
    ok = True

    # --- health
    with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
        health = json.loads(r.read())
    print(f"health: device={health['device']} limits={health['limits']} "
          f"enforce_limits={health['enforce_limits']} strategy={health['default_strategy']}")
    print(f"  knobs: {', '.join(health['knobs'])}")

    # --- 1) main path + cross-check against the offline golden
    st, hdr, res = post(body)
    print(f"\nPOST /api/evaluate (32 states) -> {st}")
    print(f"  schema_version={res.get('schema_version')} states={len(res.get('states', []))}")
    ex = res["execution"]
    print(f"  execution: strategy={ex['strategy']} forwards={ex['forwards']} "
          f"forward_passes={ex['forward_passes']} paths={ex['candidate_paths']} "
          f"prefix_sharing={ex['prefix_sharing']} wall={ex['wall_s']:.2f}s")
    print("  response headers: " + "  ".join(f"{k}={hdr.get(k)}" for k in sorted(hdr)
                                             if k.lower().startswith("x-jevinf")))
    try:
        golden = json.loads(open(GOLDEN).read())
        cmp = compare(golden, res)
        print(f"  cross-check vs offline golden: questions={cmp['questions']} "
              f"argmax agreement={cmp['argmax_agreement']*100:.2f}% "
              f"TV median={cmp['tv']['median']:.2e}")
        if cmp["argmax_agreement"] < 1.0:
            ok = False
            print(f"  ✗ mismatches: {cmp['flips'][:3]}")
    except FileNotFoundError:
        print(f"  (skipping the golden cross-check: missing {GOLDEN})")

    # --- 2) switch the strategy header: the answer semantics must not change
    st2, h2, res2 = post(body, {"x-jevinf-strategy": "two_stage"})
    cmp2 = compare(res, res2)
    print(f"\nknob x-jevinf-strategy=two_stage -> {st2} "
          f"(strategy={h2.get('x-jevinf-strategy')} forwards={h2.get('x-jevinf-forwards')})")
    print(f"  cross-check vs the default setting: argmax agreement={cmp2['argmax_agreement']*100:.2f}%")
    if cmp2["argmax_agreement"] < 1.0:
        ok = False

    # --- 3) contract boundaries
    cases = [
        ("state with one extra field", 400,
         {"states": [{"id": "s", "state": "x", "questions": {"q": {"type": "boolean",
                                                                  "instructions": "i"}},
                      "extra": 1}]}),
        ("illegal question type", 400,
         {"states": [{"id": "s", "state": "x", "questions": {"q": {"type": "nope",
                                                                  "instructions": "i"}}}]}),
        ("more than 32 states", 413,
         {"states": [{"id": f"s{i}", "state": "x",
                      "questions": {"q": {"type": "boolean", "instructions": "i"}}}
                     for i in range(33)]}),
        ("more than 256 paths", 413,
         # A single question allows at most 255 candidates (validator cap), so going over 256
         # takes two questions.
         {"states": [{"id": "s", "state": "x", "questions": {
             "big": {"type": "choice", "instructions": "i",
                     "criteria": {f"c{i}": "d" for i in range(255)}},
             "small": {"type": "choice", "instructions": "i",
                       "criteria": {f"k{i}": "d" for i in range(2)}},
         }}]}),
        ("body over 2MiB", 413, {"states": [{"id": "s", "state": "x" * (3 * 1024 * 1024),
                                             "questions": {"q": {"type": "boolean",
                                                                 "instructions": "i"}}}]}),
    ]
    print("\nContract boundaries:")
    for name, want, bad in cases:
        st3, _, r3 = post(json.dumps(bad).encode())
        good = st3 == want
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}: HTTP {st3} (expected {want}) "
              f"{r3.get('error', '')}")

    st4, _, r4 = post(b"not json")
    ok &= st4 == 400
    print(f"  {'✓' if st4 == 400 else '✗'} invalid JSON: HTTP {st4} {r4.get('error', '')}")

    st5, _, r5 = post(body, {"x-jevinf-strategy": "nope"})
    ok &= st5 == 400
    print(f"  {'✓' if st5 == 400 else '✗'} illegal strategy header: HTTP {st5} "
          f"{r5.get('error', '')}")

    print("\nVerdict:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
