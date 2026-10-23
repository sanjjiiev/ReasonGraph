/**
 * rerun.js  —  Step-Edit Re-run UI
 * ===================================
 * Loaded by index.html.
 *
 * How it works
 * ------------
 * 1.  After a CoT graph is rendered, the user can click any node in the
 *     Mermaid SVG (or pick one from a dropdown).
 * 2.  An inline editor pops up pre-filled with the step's current text.
 * 3.  The user edits the text and clicks "Re-run from here".
 * 4.  We POST to /api/rerun; on success we render the new graph side-by-side
 *     with the original and show a "changed" / "unchanged" badge.
 *
 * Dependencies
 * ------------
 * - window.lastGraphData   {nodes, edges}  — set by the main UI after /process
 * - window.mermaid                          — global Mermaid instance
 */

(function () {
  "use strict";

  /* ── Constants ────────────────────────────────────────────────────────── */
  const DEFAULT_MODEL      = "tinyllama:1.1b";
  const DEFAULT_OLLAMA_URL = "http://localhost:11434";

  /* ── State ────────────────────────────────────────────────────────────── */
  let _question   = "";   // captured from the question textarea
  let _editingId  = null; // id of the node being edited

  /* ── Inject the panel HTML once the DOM is ready ─────────────────────── */
  document.addEventListener("DOMContentLoaded", _injectPanel);

  function _injectPanel() {
    const css = `
      /* ── Rerun panel ───────────────────────────────────────── */
      #rerun-panel {
        margin-top: 24px;
        border: 1px solid #d1d5db;
        border-radius: 10px;
        background: #fff;
        overflow: hidden;
      }
      #rerun-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        background: #f3f4f6;
        cursor: pointer;
        font-weight: 600;
        font-size: 14px;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
        user-select: none;
      }
      #rerun-header .rr-toggle { font-size: 18px; }
      #rerun-body {
        padding: 16px 20px;
        display: none;
      }
      #rerun-body.open { display: block; }
      .rr-row { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }
      .rr-label { font-size: 13px; color: #6b7280; margin-bottom: 4px; }
      .rr-select, .rr-input-url, .rr-input-model {
        padding: 8px 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 13px;
        flex: 1 1 140px;
      }
      #rerun-edit-box {
        width: 100%;
        min-height: 80px;
        margin-top: 10px;
        padding: 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 13px;
        resize: vertical;
        box-sizing: border-box;
      }
      #rerun-run-btn {
        margin-top: 10px;
        padding: 9px 20px;
        background: #6366f1;
        color: #fff;
        border: none;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s;
      }
      #rerun-run-btn:hover:not(:disabled) { background: #4f46e5; }
      #rerun-run-btn:disabled { background: #a5b4fc; cursor: default; }
      #rerun-status { margin-top: 8px; font-size: 13px; color: #6b7280; }

      /* ── Comparison section ───────────────────────────────── */
      #rerun-comparison {
        margin-top: 18px;
        display: none;
      }
      #rerun-comparison.open { display: block; }
      .rr-compare-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-top: 10px;
      }
      .rr-col { border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
      .rr-col-header {
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 600;
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
      }
      .rr-col-body { padding: 12px 14px; }
      .rr-answer-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
        margin-bottom: 10px;
      }
      .rr-answer-box.original { background: #fefce8; border-color: #fde68a; }
      .rr-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
      }
      .rr-badge.changed   { background: #fee2e2; color: #b91c1c; }
      .rr-badge.unchanged { background: #d1fae5; color: #065f46; }
      .rr-badge.unknown   { background: #e0e7ff; color: #3730a3; }
      .rr-graph-area { min-height: 120px; }
    `;
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);

    const panelHtml = `
      <div id="rerun-panel">
        <div id="rerun-header" onclick="window.rerun.togglePanel()">
          <span>✏️ Step Editor &amp; Re-run</span>
          <span class="rr-toggle">▸</span>
        </div>
        <div id="rerun-body">
          <div class="rr-row">
            <div>
              <div class="rr-label">Step to edit</div>
              <select id="rr-step-select" class="rr-select">
                <option value="">— run a CoT graph first —</option>
              </select>
            </div>
            <div>
              <div class="rr-label">Ollama model</div>
              <input id="rr-model" class="rr-input-model" value="${DEFAULT_MODEL}" placeholder="${DEFAULT_MODEL}">
            </div>
            <div>
              <div class="rr-label">Ollama URL</div>
              <input id="rr-url" class="rr-input-url" value="${DEFAULT_OLLAMA_URL}" placeholder="${DEFAULT_OLLAMA_URL}">
            </div>
          </div>
          <div class="rr-label" style="margin-top:12px">Edited step text</div>
          <textarea id="rerun-edit-box" placeholder="Select a step above to pre-fill, then edit it…"></textarea>
          <button id="rerun-run-btn" onclick="window.rerun.run()">▶ Re-run from here</button>
          <div id="rerun-status"></div>

          <div id="rerun-comparison">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
              <strong style="font-size:14px">Comparison</strong>
              <span id="rr-verdict-badge" class="rr-badge unknown">—</span>
            </div>
            <div class="rr-compare-grid">
              <div class="rr-col">
                <div class="rr-col-header">📌 Original answer</div>
                <div class="rr-col-body">
                  <div id="rr-orig-answer" class="rr-answer-box original">—</div>
                  <div id="rr-orig-graph" class="rr-graph-area">
                    <em style="color:#9ca3af;font-size:12px">Original graph shown in the main panel above.</em>
                  </div>
                </div>
              </div>
              <div class="rr-col">
                <div class="rr-col-header">🔄 New answer (after edit)</div>
                <div class="rr-col-body">
                  <div id="rr-new-answer" class="rr-answer-box">—</div>
                  <div id="rr-new-graph" class="rr-graph-area mermaid-container">
                    <em style="color:#9ca3af;font-size:12px">New graph will appear here.</em>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Inject after the mermaid container section
    const anchor =
      document.getElementById("mermaid-container") ||
      document.getElementById("mermaid-diagram") ||
      document.querySelector(".output-section") ||
      document.body;

    anchor.insertAdjacentHTML("afterend", panelHtml);

    // Wire up step selector
    document.getElementById("rr-step-select").addEventListener("change", _onStepSelect);
  }

  /* ── Populate step list from the current graph ────────────────────────── */
  function _refreshStepList() {
    const select = document.getElementById("rr-step-select");
    if (!select) return;
    select.innerHTML = "";

    const gd = window.lastGraphData;
    if (!gd || !gd.nodes || gd.nodes.length === 0) {
      select.innerHTML = '<option value="">— run a CoT graph first —</option>';
      return;
    }

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "— pick a step to edit —";
    select.appendChild(placeholder);

    gd.nodes.forEach((n) => {
      if (n.id === "Q") return; // skip question node
      const opt = document.createElement("option");
      opt.value = n.id;
      opt.textContent = `${n.id}: ${n.text.replace(/\s+/g, " ").substring(0, 70)}…`;
      select.appendChild(opt);
    });
  }

  function _onStepSelect() {
    const id = document.getElementById("rr-step-select").value;
    if (!id || !window.lastGraphData) return;
    const node = window.lastGraphData.nodes.find((n) => n.id === id);
    if (node) {
      document.getElementById("rerun-edit-box").value = node.text;
      _editingId = id;
    }
  }

  /* ── Toggle panel open / close ────────────────────────────────────────── */
  function togglePanel() {
    const body   = document.getElementById("rerun-body");
    const toggle = document.querySelector("#rerun-header .rr-toggle");
    const isOpen = body.classList.toggle("open");
    toggle.textContent = isOpen ? "▾" : "▸";
    if (isOpen) _refreshStepList();
  }

  /* ── Run the re-run request ───────────────────────────────────────────── */
  async function run() {
    const editedId  = document.getElementById("rr-step-select").value;
    const newText   = document.getElementById("rerun-edit-box").value;
    const model     = document.getElementById("rr-model").value.trim()     || DEFAULT_MODEL;
    const ollamaUrl = document.getElementById("rr-url").value.trim()       || DEFAULT_OLLAMA_URL;
    const question  = (document.getElementById("question") || {}).value    || _question;

    if (!editedId) { alert("Please select a step to edit."); return; }
    if (!newText.trim()) { alert("Edited text must not be empty."); return; }
    if (!window.lastGraphData) { alert("Please run a CoT graph first."); return; }

    const btn    = document.getElementById("rerun-run-btn");
    const status = document.getElementById("rerun-status");
    btn.disabled = true;
    btn.textContent = "Running…";
    status.textContent = "Calling Ollama, please wait…";

    try {
      const resp = await fetch("/api/rerun", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nodes:      window.lastGraphData.nodes,
          edited_id:  editedId,
          new_text:   newText,
          question:   question,
          model:      model,
          ollama_url: ollamaUrl,
        }),
      });

      const result = await resp.json();

      if (!result.success) {
        status.textContent = "Error: " + result.error;
        return;
      }

      status.textContent = "Done!";
      _showComparison(result);
    } catch (err) {
      status.textContent = "Network error: " + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = "▶ Re-run from here";
    }
  }

  /* ── Render comparison section ────────────────────────────────────────── */
  function _showComparison(result) {
    // Answers
    document.getElementById("rr-orig-answer").textContent =
      result.original_answer || "(none)";
    document.getElementById("rr-new-answer").textContent =
      result.new_answer || "(none)";

    // Badge
    const badge = document.getElementById("rr-verdict-badge");
    badge.className = `rr-badge ${result.verdict}`;
    badge.textContent = result.verdict === "changed"
      ? "✗ Answer changed"
      : result.verdict === "unchanged"
      ? "✔ Answer unchanged"
      : "? Unknown";

    // New graph
    const newGraphDiv = document.getElementById("rr-new-graph");
    const mermaidCode = _nodesToMermaid(result.new_nodes, result.new_edges);
    newGraphDiv.innerHTML =
      `<div class="mermaid">${mermaidCode}</div>`;

    if (window.mermaid) {
      try {
        window.mermaid.run({ nodes: newGraphDiv.querySelectorAll(".mermaid") });
      } catch (_) {
        // Mermaid 9 fallback
        window.mermaid.init(undefined, newGraphDiv.querySelectorAll(".mermaid"));
      }
    }

    document.getElementById("rerun-comparison").classList.add("open");
  }

  /* ── Convert node/edge list to Mermaid flowchart code ─────────────────── */
  function _nodesToMermaid(nodes, edges) {
    if (!nodes || nodes.length === 0) return "flowchart TD\n  A[No nodes]";

    const lines = ["flowchart TD"];
    nodes.forEach((n) => {
      const label = (n.text || n.id).replace(/"/g, "'").replace(/\n/g, " ").substring(0, 80);
      lines.push(`  ${n.id}["${label}"]`);
    });
    (edges || []).forEach((e) => {
      lines.push(`  ${e.source} --> ${e.target}`);
    });
    return lines.join("\n");
  }

  /* ── Public API ───────────────────────────────────────────────────────── */
  window.rerun = { togglePanel, run, refreshStepList: _refreshStepList };

  /* ── Auto-refresh step list when graph changes ────────────────────────── */
  const _origSetLastGraphData = Object.getOwnPropertyDescriptor(window, "lastGraphData");
  let _gd = null;
  Object.defineProperty(window, "lastGraphData", {
    get() { return _gd; },
    set(v) {
      _gd = v;
      _refreshStepList();
      // reset comparison
      const cmp = document.getElementById("rerun-comparison");
      if (cmp) cmp.classList.remove("open");
    },
    configurable: true,
  });
})();
