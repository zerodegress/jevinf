"""Turn an upstream payload into an execution plan that can share prefixes.

Core technique: every segmented token id is reverse-derived from the `leaf_tokens` that
upstream `prepare_examples` produces, then cross-validated against an f-string version, and
any mismatch raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import upstream


@dataclass
class QuestionPlan:
    ex_id: str
    state_id: str
    qid: str
    qtype: str
    candidate_ids: list[str]
    question_seg: list[int]
    suffix_ids: list[list[int]]  # one segment per candidate (including eos)
    prefix_len: int  # state segment + question segment
    path_lengths: list[int]

    @property
    def n_paths(self) -> int:
        return len(self.suffix_ids)


@dataclass
class Plan:
    state_ids: list[str]
    state_segments: list[list[int]]
    questions: list[QuestionPlan]
    payload: dict[str, Any]

    def state_rows(self) -> list[list[int]]:
        return self.state_segments


def build_plan(payload: dict[str, Any], tokenizer, max_length: int) -> Plan:
    states = upstream.validate_request(payload)  # the same validator as upstream
    examples = upstream.prepare_examples(payload, tokenizer, max_length)
    eos = tokenizer.eos_token_id

    state_ids: list[str] = []
    seg_of: dict[str, list[int]] = {}
    for row in states:
        if row["id"] not in seg_of:
            seg_of[row["id"]] = tokenizer.encode(
                f"State:\n{row['state']}\n", add_special_tokens=False
            )
            state_ids.append(row["id"])
    qmap = {row["id"]: row["questions"] for row in states}

    questions: list[QuestionPlan] = []
    for ex in examples:
        texts = ex["candidate_texts"]
        sufs = [
            tokenizer.encode(f"Candidate:\n{t}\nDecision:", add_special_tokens=False) + [eos]
            for t in texts
        ]
        first = ex["leaf_tokens"][0]
        prefix = first[: len(first) - len(sufs[0])]

        # self-check 1: every leaf must equal exactly prefix + suffix
        for leaf, suf in zip(ex["leaf_tokens"], sufs):
            if leaf != prefix + suf:
                raise AssertionError(f"{ex['id']}: reverse-derived prefix does not match leaf_tokens")

        # self-check 2: the state segment must be the f-string encode result of the given state
        sseg = seg_of[ex["state_id"]]
        if prefix[: len(sseg)] != sseg:
            raise AssertionError(f"{ex['id']}: state segment does not match the f-string encode")
        qseg = prefix[len(sseg) :]

        # self-check 3: the question segment must equal the upstream f-string encode result (including the appended boolean criteria lines)
        q = qmap[ex["state_id"]][ex["qid"]]
        expect = f"Question type: {ex['type']}\nQuestion:\n{q['instructions']}\n"
        if ex["type"] == "boolean" and "criteria" in q:
            for key, label in (("false", "False"), ("true", "True")):
                if key in q["criteria"]:
                    expect += f"{label} criterion: {q['criteria'][key]}\n"
        if qseg != tokenizer.encode(expect, add_special_tokens=False):
            raise AssertionError(f"{ex['id']}: question segment does not match the f-string encode")

        questions.append(
            QuestionPlan(
                ex_id=ex["id"],
                state_id=ex["state_id"],
                qid=ex["qid"],
                qtype=ex["type"],
                candidate_ids=list(ex["candidate_ids"]),
                question_seg=qseg,
                suffix_ids=sufs,
                prefix_len=len(sseg) + len(qseg),
                path_lengths=[len(sseg) + len(qseg) + len(s) for s in sufs],
            )
        )

    return Plan(
        state_ids=state_ids,
        state_segments=[seg_of[s] for s in state_ids],
        questions=questions,
        payload=payload,
    )
