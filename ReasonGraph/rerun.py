"""
rerun.py — Step-Edit Re-run Engine
===================================
CoT-only. No imports from app.py.

Public API
----------
rerun_from(nodes, edited_id, new_text, question, model, ollama_url)
    -> RerunResult
"""
from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RerunNode:
    """A single reasoning step node."""
    id: str         # "Q", "S1", "S2", ..., "A"
    text: str


@dataclass
class RerunResult:
    """Outcome of a re-run operation."""
    success: bool
    error: Optional[str] = None

    # Original answer (extracted from the untouched nodes)
    original_answer: Optional[str] = None

    # New answer after regeneration
    new_answer: Optional[str] = None

    # Badge: "changed" | "unchanged" | "unknown"
    verdict: str = "unknown"

    # All new nodes (kept + edited + regenerated) in order
    new_nodes: List[RerunNode] = field(default_factory=list)

    # Edges for the new graph
    new_edges: List[dict] = field(default_factory=list)

    # Raw text returned by the model for the continuation
    raw_continuation: str = ""


# ── Ollama helper ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, ollama_url: str) -> str:
    """POST to Ollama generate endpoint. Returns the assistant text."""
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


# ── CoT XML parser (standalone, no app.py import) ────────────────────────────

def _parse_steps_and_answer(text: str) -> tuple[list[str], Optional[str]]:
    """
    Parse <step number="N">…</step> and <answer>…</answer> from model output.
    Returns (list_of_step_texts, answer_or_None).
    """
    step_pattern = r'<step\s+number="\d+"\s*>(.*?)</step>'
    steps = [m.group(1).strip() for m in re.finditer(step_pattern, text, re.DOTALL)]

    # Fallback: plain numbered lines
    if not steps:
        for m in re.finditer(r'^\s*(?:step\s*)?(\d+)\s*[.):](.+)$', text,
                             re.IGNORECASE | re.MULTILINE):
            steps.append(m.group(2).strip())

    answer_m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    answer = answer_m.group(1).strip() if answer_m else None

    if not answer:
        ans_m = re.search(
            r'(?:final\s+)?answer\s*:\s*(.+?)(?=\n\s*\n|\Z)',
            text, re.IGNORECASE | re.DOTALL,
        )
        answer = ans_m.group(1).strip() if ans_m else None

    return steps, answer


def _extract_answer_from_nodes(nodes: List[dict]) -> Optional[str]:
    """Pull the answer out of the flat node list (id == 'A')."""
    for n in nodes:
        if n.get("id") == "A":
            return n.get("text", "")
    return None


# ── Prompt builder ────────────────────────────────────────────────────────────

_COT_CONTINUATION_PROMPT = """\
You are a logical reasoning assistant.

Question: {question}

The user has already worked through part of the reasoning. Some early steps \
are given below — do NOT repeat or modify them. Starting from the last step \
shown, CONTINUE the reasoning until you reach a final answer.

{kept_steps_block}

Now continue the reasoning from here. Format each new step as:
<step number="N">
[content]
</step>

When you reach the final conclusion, write:
<answer>
[final answer]
</answer>

Continue (step {next_number} onwards):"""


def _build_prompt(question: str, kept_nodes: List[dict], next_step_number: int) -> str:
    lines = []
    for n in kept_nodes:
        if n["id"] == "Q":
            continue  # question is already in the prompt header
        if n["id"] == "A":
            continue
        lines.append(
            f'<step number="{n["id"].lstrip("S") if n["id"].startswith("S") else n["id"]}">\n'
            f'{n["text"]}\n</step>'
        )
    kept_block = "\n\n".join(lines) if lines else "(no prior steps — start from step 1)"
    return _COT_CONTINUATION_PROMPT.format(
        question=question,
        kept_steps_block=kept_block,
        next_number=next_step_number,
    )


# ── Core public function ──────────────────────────────────────────────────────

def rerun_from(
    nodes: List[dict],
    edited_id: str,
    new_text: str,
    question: str,
    model: str = "tinyllama:1.1b",
    ollama_url: str = "http://localhost:11434",
) -> RerunResult:
    """
    Regenerate CoT reasoning from an edited step onwards.

    Parameters
    ----------
    nodes       : flat list of {id, text} dicts (Q, S1, S2, …, A)
    edited_id   : id of the step the user changed (e.g. "S2")
    new_text    : the replacement text for that step
    question    : the original question
    model       : Ollama model to use for continuation
    ollama_url  : base URL of the Ollama server

    Returns
    -------
    RerunResult with new_nodes, new_edges, original_answer, new_answer, verdict
    """
    # ── Validate edited_id ────────────────────────────────────────────────────
    all_ids = [n["id"] for n in nodes]
    if edited_id not in all_ids:
        return RerunResult(
            success=False,
            error=f"Node id '{edited_id}' not found. Known ids: {all_ids}",
        )

    # ── Reject empty text ─────────────────────────────────────────────────────
    if not new_text or not new_text.strip():
        return RerunResult(
            success=False,
            error="new_text must not be empty.",
        )

    # ── Capture original answer ───────────────────────────────────────────────
    original_answer = _extract_answer_from_nodes(nodes)

    # ── Build ordered list up to and including the edited step ───────────────
    ordered_ids = [n["id"] for n in nodes]

    # Determine step ordering: Q < S1 < S2 … < A
    def _sort_key(nid: str) -> int:
        if nid == "Q":
            return 0
        if nid == "A":
            return 10_000
        try:
            return int(nid.lstrip("S"))
        except ValueError:
            return 5_000

    ordered_nodes = sorted(nodes, key=lambda n: _sort_key(n["id"]))
    edited_idx = next(i for i, n in enumerate(ordered_nodes) if n["id"] == edited_id)

    # Kept = Q through (and including) the edited step (with new text)
    kept = []
    for i, node in enumerate(ordered_nodes):
        if node["id"] == "A":
            continue  # skip original answer node
        if i < edited_idx:
            kept.append({"id": node["id"], "text": node["text"]})
        elif i == edited_idx:
            kept.append({"id": node["id"], "text": new_text.strip()})
        else:
            break  # discard later steps

    # ── Determine next step number ────────────────────────────────────────────
    step_nums = [
        int(n["id"].lstrip("S"))
        for n in kept
        if n["id"].startswith("S") and n["id"][1:].isdigit()
    ]
    next_step_number = (max(step_nums) + 1) if step_nums else 1

    # ── Build prompt and call model ───────────────────────────────────────────
    prompt = _build_prompt(question, kept, next_step_number)
    logger.info(
        "rerun_from: kept=%s edited=%s next_step=%d model=%s",
        [n["id"] for n in kept], edited_id, next_step_number, model,
    )

    try:
        continuation = _call_ollama(prompt, model, ollama_url)
    except Exception as exc:
        return RerunResult(
            success=False,
            error=f"Ollama call failed: {exc}",
            original_answer=original_answer,
        )

    # ── Parse continuation ────────────────────────────────────────────────────
    new_step_texts, new_answer = _parse_steps_and_answer(continuation)

    # ── Assemble new node list ────────────────────────────────────────────────
    new_nodes: List[RerunNode] = []
    for n in kept:
        new_nodes.append(RerunNode(id=n["id"], text=n["text"]))

    for i, step_text in enumerate(new_step_texts):
        sid = f"S{next_step_number + i}"
        new_nodes.append(RerunNode(id=sid, text=step_text))

    if new_answer:
        new_nodes.append(RerunNode(id="A", text=new_answer))

    # ── Build edges ───────────────────────────────────────────────────────────
    new_edges = []
    for i in range(1, len(new_nodes)):
        new_edges.append({
            "source": new_nodes[i - 1].id,
            "target": new_nodes[i].id,
        })

    # ── Verdict ───────────────────────────────────────────────────────────────
    if original_answer and new_answer:
        verdict = "unchanged" if original_answer.strip().lower() == new_answer.strip().lower() \
                  else "changed"
    else:
        verdict = "unknown"

    return RerunResult(
        success=True,
        original_answer=original_answer,
        new_answer=new_answer,
        verdict=verdict,
        new_nodes=new_nodes,
        new_edges=new_edges,
        raw_continuation=continuation,
    )
