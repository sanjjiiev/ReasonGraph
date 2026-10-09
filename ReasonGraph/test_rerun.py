"""
test_rerun.py — Unit + integration tests for the step-rerun engine.
=====================================================================
Run with:
    python -m pytest test_rerun.py -v

No network required for unit tests; integration tests require Ollama running
and are skipped automatically if it is not reachable.
"""
from __future__ import annotations

import json
import sys
import os
import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from rerun import rerun_from, _parse_steps_and_answer, _build_prompt, RerunResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_NODES = [
    {"id": "Q",  "text": "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?"},
    {"id": "S1", "text": "Let ball = x dollars."},
    {"id": "S2", "text": "Bat = x + 1.00, so x + (x + 1.00) = 1.10."},
    {"id": "S3", "text": "2x + 1.00 = 1.10 → 2x = 0.10 → x = 0.05."},
    {"id": "A",  "text": "The ball costs $0.05."},
]

QUESTION = "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?"


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — no network needed
# ─────────────────────────────────────────────────────────────────────────────

class TestParseStepsAndAnswer:
    def test_xml_format(self):
        text = '''
<step number="1">First step content.</step>
<step number="2">Second step.</step>
<answer>The answer is 42.</answer>
'''
        steps, answer = _parse_steps_and_answer(text)
        assert steps == ["First step content.", "Second step."]
        assert answer == "The answer is 42."

    def test_plain_numbered_fallback(self):
        text = "1. Do this first.\n2. Then do that.\nAnswer: forty-two"
        steps, answer = _parse_steps_and_answer(text)
        assert len(steps) >= 1
        assert answer is not None and "forty-two" in answer.lower()

    def test_no_answer_returns_none(self):
        text = '<step number="1">Only one step, no answer tag.</step>'
        steps, answer = _parse_steps_and_answer(text)
        assert len(steps) == 1
        assert answer is None


class TestBuildPrompt:
    def test_includes_question(self):
        kept = [{"id": "Q", "text": QUESTION}, {"id": "S1", "text": "Let x = ball price."}]
        prompt = _build_prompt(QUESTION, kept, 2)
        assert QUESTION in prompt
        assert "Let x = ball price." in prompt
        assert "step 2" in prompt.lower()

    def test_no_q_node_in_kept_steps(self):
        kept = [{"id": "Q", "text": QUESTION}]
        prompt = _build_prompt(QUESTION, kept, 1)
        # Q node text should NOT appear in the kept block (it's in the header)
        assert "no prior steps" in prompt

    def test_next_number_used(self):
        kept = [
            {"id": "Q",  "text": QUESTION},
            {"id": "S1", "text": "Step 1 text."},
            {"id": "S2", "text": "Step 2 text."},
        ]
        prompt = _build_prompt(QUESTION, kept, 3)
        assert "step 3" in prompt.lower()


class TestRerunFromValidation:
    """Tests that don't call Ollama — just validation paths."""

    def _make_result(self, **kwargs) -> RerunResult:
        """Call rerun_from with a bogus ollama_url so the network fails fast,
        but test validation before the network call."""
        return rerun_from(
            ollama_url="http://127.0.0.1:19999",  # nothing running here
            **kwargs,
        )

    def test_unknown_id_returns_error(self):
        result = rerun_from(
            nodes=SAMPLE_NODES,
            edited_id="S99",
            new_text="Some text",
            question=QUESTION,
            ollama_url="http://127.0.0.1:19999",
        )
        assert result.success is False
        assert "S99" in result.error

    def test_empty_text_returns_error(self):
        result = rerun_from(
            nodes=SAMPLE_NODES,
            edited_id="S1",
            new_text="   ",
            question=QUESTION,
            ollama_url="http://127.0.0.1:19999",
        )
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_edit_last_step_validation_passes(self):
        """Editing the last step (S3) is valid; it will fail at network level."""
        result = rerun_from(
            nodes=SAMPLE_NODES,
            edited_id="S3",
            new_text="Revised last step.",
            question=QUESTION,
            ollama_url="http://127.0.0.1:19999",
        )
        # It may succeed=False due to network, but the error must be about Ollama
        if not result.success:
            assert "Ollama" in result.error or "Connection" in result.error

    def test_edit_first_step_validation_passes(self):
        """Editing the first step (S1) should also be valid."""
        result = rerun_from(
            nodes=SAMPLE_NODES,
            edited_id="S1",
            new_text="Let ball = y dollars.",
            question=QUESTION,
            ollama_url="http://127.0.0.1:19999",
        )
        if not result.success:
            assert "Ollama" in result.error or "Connection" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — require Ollama running with tinyllama:1.1b
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        import requests as _r
        r = _r.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


OLLAMA_MARK = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running — skipping integration tests",
)


@OLLAMA_MARK
class TestRerunIntegration:
    """
    Run 10 question/edit scenarios and report how often the answer changes.
    Requires: ollama serve && ollama pull tinyllama:1.1b
    """

    SCENARIOS = [
        # (question, nodes_partial, edit_step_id, new_text, should_change)
        (
            "What is 2 + 2?",
            [
                {"id": "Q",  "text": "What is 2 + 2?"},
                {"id": "S1", "text": "2 + 2 = 4."},
                {"id": "A",  "text": "4"},
            ],
            "S1", "Actually 2 + 2 could be written as 2 × 2 = 4.", False,
        ),
        (
            "What is 3 × 5?",
            [
                {"id": "Q",  "text": "What is 3 × 5?"},
                {"id": "S1", "text": "3 × 5 = 15."},
                {"id": "A",  "text": "15"},
            ],
            "S1", "I think 3 × 5 is 100.", True,
        ),
        (
            "Is 7 a prime number?",
            [
                {"id": "Q",  "text": "Is 7 a prime number?"},
                {"id": "S1", "text": "7 is only divisible by 1 and 7."},
                {"id": "S2", "text": "So 7 is prime."},
                {"id": "A",  "text": "Yes, 7 is prime."},
            ],
            "S1", "7 is divisible by 1, 7, and also 14.", True,
        ),
        (
            "What is the boiling point of water in Celsius?",
            [
                {"id": "Q",  "text": "What is the boiling point of water in Celsius?"},
                {"id": "S1", "text": "Water boils at 100°C at standard pressure."},
                {"id": "A",  "text": "100°C"},
            ],
            "S1", "Water boils at 37°C.", True,
        ),
        (
            "How many days in a week?",
            [
                {"id": "Q",  "text": "How many days in a week?"},
                {"id": "S1", "text": "There are 7 days: Mon, Tue, Wed, Thu, Fri, Sat, Sun."},
                {"id": "A",  "text": "7"},
            ],
            "S1", "There are 7 days in a week.", False,
        ),
        (
            "What is 10 - 4?",
            [
                {"id": "Q",  "text": "What is 10 - 4?"},
                {"id": "S1", "text": "10 - 4 = 6."},
                {"id": "A",  "text": "6"},
            ],
            "S1", "10 - 4 = 99.", True,
        ),
        (
            "Is the earth flat?",
            [
                {"id": "Q",  "text": "Is the earth flat?"},
                {"id": "S1", "text": "No, the Earth is an oblate spheroid."},
                {"id": "A",  "text": "No."},
            ],
            "S1", "Yes, the Earth is completely flat.", True,
        ),
        (
            "What is 100 / 4?",
            [
                {"id": "Q",  "text": "What is 100 / 4?"},
                {"id": "S1", "text": "100 / 4 = 25."},
                {"id": "A",  "text": "25"},
            ],
            "S1", "100 / 4 = 25.", False,
        ),
        (
            "How many seconds in a minute?",
            [
                {"id": "Q",  "text": "How many seconds in a minute?"},
                {"id": "S1", "text": "1 minute = 60 seconds."},
                {"id": "A",  "text": "60"},
            ],
            "S1", "1 minute = 60 seconds.", False,
        ),
        (
            "What is the square root of 16?",
            [
                {"id": "Q",  "text": "What is the square root of 16?"},
                {"id": "S1", "text": "√16 = 4 because 4 × 4 = 16."},
                {"id": "A",  "text": "4"},
            ],
            "S1", "√16 = 999 because I said so.", True,
        ),
    ]

    def test_scenarios(self):
        results = []
        for q, nodes, edit_id, new_text, expected_change in self.SCENARIOS:
            r = rerun_from(
                nodes=nodes,
                edited_id=edit_id,
                new_text=new_text,
                question=q,
                model="tinyllama:1.1b",
            )
            changed = r.verdict == "changed"
            match   = changed == expected_change
            results.append({
                "question":        q,
                "edit_id":         edit_id,
                "expected_change": expected_change,
                "actual_verdict":  r.verdict,
                "match":           match,
                "new_answer":      r.new_answer,
                "error":           r.error,
            })
            print(
                f"[{'✔' if match else '✗'}] Q={q[:40]!r}  "
                f"expected_change={expected_change}  verdict={r.verdict}"
            )

        matches   = sum(1 for x in results if x["match"])
        total     = len(results)
        successes = sum(1 for x in results if not x["error"])

        print(f"\n{'='*60}")
        print(f"Results: {matches}/{total} verdicts matched expectation")
        print(f"         {successes}/{total} model calls succeeded")
        print(f"{'='*60}")
        print(json.dumps(results, indent=2))

        # Soft assertion: at least half should match
        assert matches >= total // 2, (
            f"Too few verdicts matched: {matches}/{total}"
        )
