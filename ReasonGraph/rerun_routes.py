"""
rerun_routes.py — Flask Blueprint for the step-rerun API.
==========================================================
Registered in app.py with one line:
    from rerun_routes import rerun_bp
    app.register_blueprint(rerun_bp)
"""
from __future__ import annotations

import logging
from flask import Blueprint, request, jsonify
from rerun import rerun_from, RerunResult

logger = logging.getLogger(__name__)

rerun_bp = Blueprint("rerun", __name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL      = "tinyllama:1.1b"


def _result_to_dict(r: RerunResult) -> dict:
    return {
        "success":          r.success,
        "error":            r.error,
        "original_answer":  r.original_answer,
        "new_answer":       r.new_answer,
        "verdict":          r.verdict,
        "raw_continuation": r.raw_continuation,
        "new_nodes": [{"id": n.id, "text": n.text} for n in r.new_nodes],
        "new_edges": r.new_edges,
    }


@rerun_bp.route("/api/rerun", methods=["POST"])
def api_rerun():
    """
    POST /api/rerun
    ---------------
    Body (JSON):
        {
            "nodes":     [{id, text}, ...],   # full current CoT node list
            "edited_id": "S2",                # id of the edited step
            "new_text":  "...",               # replacement text
            "question":  "...",               # original question
            "model":     "tinyllama:1.1b",    # optional Ollama model
            "ollama_url":"http://...",         # optional Ollama URL
        }

    Returns (JSON):
        {
            "success":         true,
            "original_answer": "...",
            "new_answer":      "...",
            "verdict":         "changed" | "unchanged" | "unknown",
            "new_nodes":       [{id, text}, ...],
            "new_edges":       [{source, target}, ...],
            "raw_continuation":"...",
        }
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"success": False, "error": "Request body must be JSON."}), 400

    nodes      = body.get("nodes")
    edited_id  = body.get("edited_id", "")
    new_text   = body.get("new_text", "")
    question   = body.get("question", "")
    model      = body.get("model", _DEFAULT_MODEL)
    ollama_url = body.get("ollama_url", _DEFAULT_OLLAMA_URL)

    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(nodes, list) or not nodes:
        return jsonify({"success": False,
                        "error": "'nodes' must be a non-empty list."}), 400
    if not edited_id:
        return jsonify({"success": False, "error": "'edited_id' is required."}), 400
    if not question:
        return jsonify({"success": False, "error": "'question' is required."}), 400

    logger.info(
        "api_rerun: edited_id=%s model=%s nodes=%d",
        edited_id, model, len(nodes),
    )

    result = rerun_from(
        nodes=nodes,
        edited_id=edited_id,
        new_text=new_text,
        question=question,
        model=model,
        ollama_url=ollama_url,
    )

    status_code = 200 if result.success else 422
    return jsonify(_result_to_dict(result)), status_code
