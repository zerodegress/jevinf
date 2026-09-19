"""Hit the local service with the **official typesafe-sdk** — the strictest Jev conformance check.

The criteria chain:
  1. The official SDK validates the response with its own pydantic models
     (`_schemas/models.py`, generated from the official OpenAPI). If it does not raise
     `TypeSafeAPIResponseValidationError` = the wire layer passes.
  2. The SDK's typed accessors (`.nouls` / `.choices` / `.scores`) can read out the fields.
  3. The same state goes through both `/v1/systemone` (translation layer) and `/api/evaluate`
     (engine native), compared question by question — proving the translation layer did not
     change the answer semantics.
  4. Error contract: Jev specifies that validation failure is 422, so the SDK should raise
     TypeSafeUnprocessableEntityError.

Usage: uv run python scripts/jev_conformance.py --base http://127.0.0.1:8226
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ap = argparse.ArgumentParser(description="Jev conformance check")
_ap.add_argument("--base", default="http://127.0.0.1:8226")
_ap.add_argument("--api-key", default="local-test-key")
ARGS = _ap.parse_args()

BASE = ARGS.base
os.environ["TYPESAFE_BASE_URL"] = BASE
os.environ.setdefault("TYPESAFE_API_KEY", ARGS.api_key)

import urllib.request  # noqa: E402

try:
    with urllib.request.urlopen(f"{BASE}/health", timeout=30) as _r:
        HEALTH = json.loads(_r.read())
except Exception as exc:  # noqa: BLE001
    sys.exit(f"/health did not answer with JSON at {BASE}: {type(exc).__name__}: {str(exc)[:200]}")
ARRANGEMENT = HEALTH.get("arrangement")
print(f"backend: architecture={HEALTH.get('architecture')} arrangement={ARRANGEMENT} "
      f"limits={HEALTH.get('limits')} temperature_default={HEALTH.get('default_temperature')}\n")

from typesafe_sdk import (  # noqa: E402
    Choice,
    Noul,
    Score,
    TypeSafeClient,
    TypeSafeUnprocessableEntityError,
)

STATE = ("Incident: error rate 3.4%, p99 latency 850ms, the last deploy was 40 minutes ago "
         "and the on-call engineer is paged. Customer says payouts have been failing for "
         "three days and nobody replied to their emails.")

QUESTIONS = {
    "billing": Noul(instructions="Is this message about billing?",
                    criteria={"true": "Charges, refunds or payouts are involved",
                              "false": "No billing component"}),
    "urgent": Noul(instructions="Does this need a reply within the hour?"),
    "team": Choice(instructions="Which team should handle it?",
                   criteria={"payments": "Payouts, refunds, charges",
                             "platform": "Latency, outages, errors",
                             "other": None}),
    "tone": Score(instructions="How frustrated is the customer?",
                  criteria=["calm", "frustrated", "very angry"]),
}

NANO_QUESTIONS = {  # The same question set, asked again in the engine-native shape
    "billing": {"type": "boolean", "instructions": "Is this message about billing?",
                "criteria": {"true": "Charges, refunds or payouts are involved",
                             "false": "No billing component"}},
    "urgent": {"type": "boolean", "instructions": "Does this need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"payments": "Payouts, refunds, charges",
                          "platform": "Latency, outages, errors",
                          "other": "other"}},
    "tone": {"type": "score", "instructions": "How frustrated is the customer?",
             "criteria": ["calm", "frustrated", "very angry"]},
}

# The **pure JSON form** of the same question set, used to bypass the SDK and hit the raw wire
# directly (the SDK's Question is a pydantic object, it cannot be json.dumps'd). The contents are
# field-for-field identical to QUESTIONS.
RAW_QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this message about billing?",
                "criteria": {"true": "Charges, refunds or payouts are involved",
                             "false": "No billing component"}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"payments": "Payouts, refunds, charges",
                          "platform": "Latency, outages, errors",
                          "other": None}},
    "tone": {"type": "score", "instructions": "How frustrated is the customer?",
             "criteria": ["calm", "frustrated", "very angry"]},
}

ok = True


def check(label: str, cond: bool, extra: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"  {'✓' if cond else '✗'} {label}{(' — ' + extra) if extra else ''}")


def post_native(payload: dict) -> dict:
    req = urllib.request.Request(f"{BASE}/api/evaluate", data=json.dumps(payload).encode(),
                                 method="POST", headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    print(f"Official SDK -> {BASE}\n")
    with TypeSafeClient() as client:
        # --- /v1/models
        models = client.models.list()
        names = [m.name for m in models.models]
        print(f"GET /v1/models: {names}")
        check("model catalog readable and contains jev-latest", "jev-latest" in names)
        for m in models.models:
            check(f"metadata complete: {m.name}",
                  bool(m.description) and bool(m.release_date), m.release_date)

        # --- system_one: all three question types mixed into a single request
        res = client.system_one(state=STATE, questions=QUESTIONS)
        print(f"\nPOST /v1/systemone ← model={res.model} usage={res.usage.model_dump()}")
        print(json.dumps(res.model_dump(), ensure_ascii=False, indent=1))
        check("SDK response validation passed (no TypeSafeAPIResponseValidationError raised)", True)
        check("noul readable", 0.0 <= res.nouls["billing"].noul <= 1.0,
              f"billing={res.nouls['billing'].noul}")
        check("noul has no confidence field", not hasattr(res.nouls["billing"], "confidence"))
        check("choice readable and carries confidence",
              res.choices["team"].choice in {"payments", "platform", "other"}
              and 0.0 <= res.choices["team"].confidence <= 1.0,
              f"team={res.choices['team'].choice} conf={res.choices['team'].confidence}")
        sp = res.scores["tone"]
        # NOTE: the SDK's **user-convenience model** declares legend/probabilities as dict[int, ...],
        # and pydantic coerces the string keys ("0"/"1"/"2") into int — the wire contract itself is
        # string keys (official OpenAPI: dict[str, ...] + "A level index, as a string key matching
        # legend").
        check("score readable, legend complete, confidence within bounds",
              0.0 <= sp.score <= 2.0 and {int(k) for k in sp.legend} == {0, 1, 2}
              and 0.0 <= sp.confidence <= 1.0,
              f"score={sp.score} legend={sp.legend} conf={sp.confidence}")
        for qid in ("team", "tone"):
            probs = (res.choices[qid] if qid == "team" else res.scores[qid]).probabilities
            check(f"{qid} probabilities sum to ~1", abs(sum(probs.values()) - 1.0) < 1e-4,
                  f"Σ={sum(probs.values()):.6f}")
        check("usage.input_tokens > 0", res.usage.input_tokens > 0,
              f"input={res.usage.input_tokens} output={res.usage.output_tokens}")

        # --- raw wire (the body never went through the SDK models): keys must be **strings**,
        # aligned with the official OpenAPI
        raw_req = urllib.request.Request(
            f"{BASE}/v1/systemone",
            data=json.dumps({"state": STATE, "model": "jev-latest",
                             "questions": RAW_QUESTIONS}).encode(),
            method="POST", headers={"content-type": "application/json",
                                    "authorization": "Bearer local-test-key"})
        with urllib.request.urlopen(raw_req, timeout=600) as r:
            raw = json.loads(r.read())
            raw_hdr = dict(r.headers)
        wire = raw["answers"]["tone"]
        check("legend/probabilities keys on the wire are strings",
              all(isinstance(k, str) for k in wire["legend"])
              and all(isinstance(k, str) for k in wire["probabilities"]),
              f"legend keys={list(wire['legend'])}")
        check("wire answer fields complete (choice has confidence/choice/probabilities)",
              set(raw["answers"]["team"]) == {"type", "choice", "confidence", "probabilities"})
        check("wire noul answer has only type/noul",
              set(raw["answers"]["billing"]) == {"type", "noul"})
        check("response header carries x-typesafe-request-id",
              bool(raw_hdr.get("x-typesafe-request-id")))

        # --- the translation layer must not change semantics: same state through the native
        # endpoint, compared
        native = post_native({"states": [{"id": "s", "state": STATE,
                                          "questions": NANO_QUESTIONS}]})
        ans = native["states"][0]["answers"]
        p_true = lambda qid: ans[qid].get("p_true", ans[qid].get("noul"))  # nanojev / decider vocabulary
        same_noul = abs(p_true("billing") - res.nouls["billing"].noul) < 1e-6
        same_urgent = abs(p_true("urgent") - res.nouls["urgent"].noul) < 1e-6
        same_choice = ans["team"]["choice"] == res.choices["team"].choice
        same_score = abs(ans["tone"]["score"] - res.scores["tone"].score) < 1e-6
        print(f"\nnative endpoint cross-check: billing {p_true('billing')} "
              f"vs {res.nouls['billing'].noul}"
              f" | team {ans['team']['choice']} vs {res.choices['team'].choice}"
              f" | tone {ans['tone']['score']} vs {res.scores['tone'].score}")
        check("noul(billing) matches the engine native", same_noul)
        check("noul(urgent) matches the engine native", same_urgent)
        check("choice matches the engine native", same_choice)
        check("score matches the engine native", same_score)

        # --- error contract
        print("\nError contract:")
        # Deliberately far past any per-question budget these backends have (thousands of tokens, not
        # "probably over 512"): the check is the error contract, and a state that happens to fit would
        # turn it into a coin flip.
        long_state = ("The grid reports blocked cells, visited cells, the current heading and "
                      "the remaining distance to the goal for the active navigation episode. ") * 80
        cases = []
        if ARRANGEMENT != "state-fork":
            # A per-question budget is a property of the three-stage and single-path families: the first
            # caps a candidate path at 512 tokens, the second caps a whole sequence. The state-fork
            # family's budget is two orders of magnitude larger, so the same request must be served
            # there instead (checked below).
            cases.append(("state far past the per-question budget (backend hard limit)",
                          dict(state=long_state, questions={"q": Noul(instructions="ok?")})))
        cases += [
            ("score with only 1 level (legal in Jev, the backend cannot serve it)",
             dict(state=STATE, questions={"q": Score(instructions="rate", criteria=["only"])})),
            ("choice with only 1 candidate",
             dict(state=STATE, questions={"q": Choice(instructions="pick",
                                                      criteria={"a": None})})),
            ("noul missing instructions",
             dict(state=STATE, questions={"q": Noul(instructions=None)})),
        ]
        for label, kwargs in cases:
            try:
                client.system_one(**kwargs)
                check(label, False, "it unexpectedly succeeded")
            except TypeSafeUnprocessableEntityError as err:
                detail = str(err)[:130].replace("\n", " ")
                check(label, True, f"422 {detail}")
            except Exception as exc:  # noqa: BLE001
                check(label, False, f"{type(exc).__name__}: {str(exc)[:130]}")

        if ARRANGEMENT == "state-fork":
            # The complement of the case skipped above: a state far beyond that path cap is legal here,
            # and the request must actually be served (not silently truncated without saying so).
            # A client with more patience than the SDK's 10 s default: this family prefills at a few
            # milliseconds per token, so a long state is a slow request rather than a wrong one.
            big_state = long_state * 12
            try:
                import time as _time

                t0 = _time.perf_counter()
                with TypeSafeClient(timeout=300.0) as patient:
                    res_long = patient.system_one(state=big_state, questions={"q": Noul(instructions="ok?")})
                elapsed = _time.perf_counter() - t0
                tokens = res_long.usage.input_tokens
                check("a state far past the 512-token path cap is served, not refused",
                      tokens > 600, f"usage.input_tokens={tokens} for {len(big_state)} characters")
                # What a default client can actually reach, measured rather than guessed.
                rate = tokens / elapsed if elapsed > 0 else 0.0
                print(f"  · {tokens} tokens in {elapsed:.1f} s ({rate:.0f} tok/s): a client with the "
                      f"SDK's default 10 s timeout reaches about {int(rate * 10)} tokens here")
            except Exception as exc:  # noqa: BLE001
                check("a state far past the 512-token path cap is served, not refused", False,
                      f"{type(exc).__name__}: {str(exc)[:130]}")

        # Everything outside the server-side contract (an illegal question type) is rejected by
        # pydantic before it ever reaches the SDK/server
        try:
            client.system_one(state=STATE, questions={"q": {"type": "bogus"}})
            check("illegal question type rejected", False, "it unexpectedly succeeded")
        except TypeSafeUnprocessableEntityError:
            check("illegal question type rejected (422)", True)
        except Exception as exc:  # noqa: BLE001
            check("illegal question type rejected", True,
                  f"{type(exc).__name__}(a client-side rejection first also counts)")

    print("\nVerdict:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
