from flask import Flask, render_template, request, jsonify
from api_base import create_api  # New import for API factory
from plain_text_reasoning import (
    create_mermaid_diagram as create_plain_diagram,
    parse_plain_text_response
)
from cot_reasoning import (
    VisualizationConfig,
    create_mermaid_diagram as create_cot_diagram,
    parse_cot_response
)
from tot_reasoning import (
    create_mermaid_diagram as create_tot_diagram,
    parse_tot_response
)
from l2m_reasoning import (
    create_mermaid_diagram as create_l2m_diagram,
    parse_l2m_response
)
from selfconsistency_reasoning import (
    create_mermaid_diagram as create_scr_diagram,
    parse_scr_response
)
from selfrefine_reasoning import (
    create_mermaid_diagram as create_srf_diagram,
    parse_selfrefine_response
)
from bs_reasoning import (
    create_mermaid_diagram as create_bs_diagram,
    parse_bs_response
)
from configs import config
import graph_evaluator
import logging
import json
import os
import uuid
from datetime import datetime, timezone


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Verification Module Integration (Feature Flagged)
VERIFY_ENABLED = True
if VERIFY_ENABLED:
    try:
        from verify_routes import verify_bp
        app.register_blueprint(verify_bp)
    except ImportError as e:
        logger.warning(f"Could not load verification module: {e}")

# ── Benchmark persistence ─────────────────────────────────────────────────────
_BENCH_DIR   = os.path.join(os.path.dirname(__file__), "benchmarks")
_BENCH_STORE = os.path.join(_BENCH_DIR, "results.json")

def _load_benchmark_results():
    """Load benchmark history from JSON file."""
    try:
        with open(_BENCH_STORE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def _save_benchmark_results(results):
    """Persist benchmark history to JSON file."""
    os.makedirs(_BENCH_DIR, exist_ok=True)
    with open(_BENCH_STORE, "w") as f:
        json.dump(results, f, indent=2)


def _extract_graph_data(parsed_result, reasoning_method: str):
    """
    Extract a flat (nodes, edges) representation from any parsed result object.
    Returns (nodes, edges) where:
      nodes = [{"id": str, "text": str}, ...]
      edges = [{"source": str, "target": str}, ...]
    """
    nodes = []
    edges = []

    try:
        if reasoning_method == 'cot':
            # CoTResponse: question, steps (CoTStep list), answer
            nodes.append({"id": "Q", "text": parsed_result.question})
            prev = "Q"
            for step in parsed_result.steps:
                sid = f"S{step.number}"
                nodes.append({"id": sid, "text": step.content})
                edges.append({"source": prev, "target": sid})
                prev = sid
            if parsed_result.answer:
                nodes.append({"id": "A", "text": parsed_result.answer})
                edges.append({"source": prev, "target": "A"})

        elif reasoning_method in ('tot', 'bs'):
            nodes.append({"id": "Q", "text": parsed_result.question})
            if parsed_result.root:
                edges.append({"source": "Q", "target": str(parsed_result.root.id)})
                
                queue = [parsed_result.root]
                while queue:
                    curr = queue.pop(0)
                    nodes.append({"id": str(curr.id), "text": str(curr.content)})
                    
                    parent_id = getattr(curr, 'parent_id', None)
                    if parent_id:
                        edges.append({"source": str(parent_id), "target": str(curr.id)})
                        
                    queue.extend(curr.children or [])
            
            if parsed_result.answer:
                nodes.append({"id": "A", "text": parsed_result.answer})
                # connect answer to all sinks
                child_ids = {e["target"] for e in edges}
                parent_ids = {e["source"] for e in edges}
                sinks = [n["id"] for n in nodes if n["id"] in parent_ids and n["id"] not in child_ids and n["id"] != "Q"]
                
                # fallback if no sinks found but root exists
                if not sinks and parsed_result.root:
                    sinks = [str(parsed_result.root.id)]
                    
                for sink in sinks:
                    edges.append({"source": sink, "target": "A"})

        elif reasoning_method == 'l2m':
            # L2MResponse: question, steps (L2MStep: question, reasoning, answer), final_answer
            nodes.append({"id": "Q", "text": parsed_result.question})
            prev = "Q"
            for i, step in enumerate(parsed_result.steps):
                sid = f"S{i+1}"
                text = f"{step.question} | {step.reasoning}"
                nodes.append({"id": sid, "text": text})
                edges.append({"source": prev, "target": sid})
                prev = sid
            if parsed_result.final_answer:
                nodes.append({"id": "A", "text": parsed_result.final_answer})
                edges.append({"source": prev, "target": "A"})

        elif reasoning_method == 'scr':
            # SCRResponse: question, paths (list of CoT paths)
            nodes.append({"id": "Q", "text": parsed_result.question})
            for pi, path in enumerate(parsed_result.paths):
                prev = "Q"
                for step in path.steps:
                    sid = f"P{pi+1}S{step.number}"
                    nodes.append({"id": sid, "text": step.content})
                    edges.append({"source": prev, "target": sid})
                    prev = sid
                if path.answer:
                    aid = f"P{pi+1}A"
                    nodes.append({"id": aid, "text": path.answer})
                    edges.append({"source": prev, "target": aid})
            if parsed_result.final_answer:
                nodes.append({"id": "FA", "text": parsed_result.final_answer})

        elif reasoning_method == 'srf':
            # SRFResponse: question, steps, answer, revision_check, revised_steps, revised_answer
            nodes.append({"id": "Q", "text": parsed_result.question})
            prev = "Q"
            for step in parsed_result.steps:
                sid = f"S{step.number}"
                nodes.append({"id": sid, "text": step.content})
                edges.append({"source": prev, "target": sid})
                prev = sid
            if parsed_result.answer:
                nodes.append({"id": "A", "text": parsed_result.answer})
                edges.append({"source": prev, "target": "A"})
                prev = "A"
            for rstep in (parsed_result.revised_steps or []):
                rsid = f"RS{rstep.number}"
                nodes.append({"id": rsid, "text": rstep.content})
                edges.append({"source": prev, "target": rsid})
                prev = rsid
            if parsed_result.revised_answer:
                nodes.append({"id": "RA", "text": parsed_result.revised_answer})
                edges.append({"source": prev, "target": "RA"})

    except Exception as exc:
        logger.warning(f"Graph extraction partially failed for method={reasoning_method}: {exc}")

    return nodes, edges



@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/index.html')
def index_direct():
    """Directly render the main page when accessed via index.html"""
    return render_template('index.html')

@app.route('/index_cn.html')
def index_cn():
    """Render the Chinese version of the main page"""
    return render_template('index_cn.html')

@app.route('/config')
def get_config():
    """Get initial configuration"""
    return jsonify(config.get_initial_values())

@app.route('/method-config/<method_id>')
def get_method_config(method_id):
    """Get configuration for specific method"""
    method_config = config.get_method_config(method_id)
    if method_config:
        return jsonify(method_config)
    return jsonify({"error": "Method not found"}), 404

@app.route('/provider-api-key/<provider>')
def get_provider_api_key(provider):
    """Get default API key for specific provider"""
    try:
        api_key = config.general.get_default_api_key(provider)
        return jsonify({
            'success': True,
            'api_key': api_key
        })
    except Exception as e:
        logger.error(f"Error getting API key for provider {provider}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/save-api-key', methods=['POST'])
def save_api_key():
    """Save API key for a provider in memory only (no file storage)"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        provider = data.get('provider')
        api_key = data.get('api_key')

        if not provider or not api_key:
            return jsonify({
                'success': False,
                'error': 'Provider and API key are required'
            }), 400

        # Update API key in config (this updates the in-memory API keys only)
        config.general.provider_api_keys[provider] = api_key
        logger.info(f"Saved API key for provider: {provider} (in memory only)")

        return jsonify({
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error saving API key: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/select-method', methods=['POST'])
def select_method():
    """Let the model select the most appropriate reasoning method"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Extract parameters
        api_key = data.get('api_key')
        provider = data.get('provider', 'anthropic')
        model = data.get('model')
        question = data.get('question')

        if not all([api_key, model, question]):
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400

        # Create the selection prompt
        methods = config.methods
        prompt = f"""Given this question: "{question}"

Please select the most appropriate reasoning method from the following options to solve it:

{chr(10).join(f'- {method_id}: {config.name}' for method_id, config in methods.items())}

Consider the characteristics of each method and the nature of the question.
Output your selection in exactly this format:
<selected_method>method_id</selected_method>
where method_id is strictly one of: {', '.join(methods.keys())}.
Do not use the method or words that are not in {', '.join(methods.keys())}."""

        # Get model's selection
        try:
            api = create_api(provider, api_key, model)
            response = api.generate_response(prompt, max_tokens=100)
            
            # Extract method ID using basic string parsing
            import re
            match = re.search(r'<selected_method>(\w+)</selected_method>', response)
            if match and match.group(1) in methods:
                selected_method = match.group(1)
                return jsonify({
                    'success': True,
                    'selected_method': selected_method,
                    'raw_response': response
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Invalid method selection in response'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'API call failed: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in method selection: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process', methods=['POST'])
def process():
    """Process the reasoning request"""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Extract parameters
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key is required'
            }), 400

        question = data.get('question')
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        # Get optional parameters with defaults
        provider = data.get('provider', 'anthropic')  # New parameter for provider
        model = data.get('model', config.general.available_models[0])
        max_tokens = int(data.get('max_tokens', config.general.max_tokens))
        prompt_format = data.get('prompt_format')
        chars_per_line = int(data.get('chars_per_line', config.general.chars_per_line))
        max_lines = int(data.get('max_lines', config.general.max_lines))
        reasoning_method = data.get('reasoning_method', 'cot')
        
        # Initialize API with factory function
        try:
            api = create_api(provider, api_key, model)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to initialize API: {str(e)}'
            }), 400
        
        # Get model response
        logger.info(f"Generating response for question using {provider} {model}")
        try:
            raw_response = api.generate_response(
                question,
                max_tokens=max_tokens,
                prompt_format=prompt_format
            )
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'API call failed: {str(e)}'
            }), 500
        
        # Create visualization config
        viz_config = VisualizationConfig(
            max_chars_per_line=chars_per_line,
            max_lines=max_lines
        )
        
        # Generate visualization based on reasoning method
        visualization = None
        try:
            if reasoning_method == 'cot':
                result = parse_cot_response(raw_response, question)
                visualization = create_cot_diagram(result, viz_config)
            elif reasoning_method == 'tot':
                result = parse_tot_response(raw_response, question)
                visualization = create_tot_diagram(result, viz_config)
            elif reasoning_method == 'l2m':
                result = parse_l2m_response(raw_response, question)
                visualization = create_l2m_diagram(result, viz_config)
            elif reasoning_method == 'scr':
                result = parse_scr_response(raw_response, question)
                visualization = create_scr_diagram(result, viz_config)
            elif reasoning_method == 'srf':
                result = parse_selfrefine_response(raw_response, question)
                visualization = create_srf_diagram(result, viz_config)
            elif reasoning_method == 'bs':
                result = parse_bs_response(raw_response, question)
                visualization = create_bs_diagram(result, viz_config)
            elif reasoning_method == 'plain':
                parse_plain_text_response(raw_response, question)
                visualization = None
                
            logger.info("Successfully generated visualization")
        except Exception as viz_error:
            logger.error(f"Visualization generation failed: {str(viz_error)}")
            # Continue without visualization
        
        # Extract graph data for evaluation panel
        graph_nodes, graph_edges = [], []
        if result is not None and reasoning_method != 'plain':
            try:
                graph_nodes, graph_edges = _extract_graph_data(result, reasoning_method)
            except Exception as ge:
                logger.warning(f"Graph data extraction failed: {ge}")

        # Return successful response
        return jsonify({
            'success': True,
            'raw_output': raw_response,
            'visualization': visualization,
            'graph_data': {
                'nodes': graph_nodes,
                'edges': graph_edges,
            }
        })

        
    except Exception as e:
        # Log the error and return error response
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ── Graph Evaluation Endpoint ────────────────────────────────────────────────

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    POST /evaluate
    Body: {"nodes": [{"id":..., "text":...}], "edges": [{"source":..., "target":...}]}
    Returns RQI metrics as JSON.
    """
    try:
        data = request.json or {}
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        if not nodes:
            return jsonify({'success': False, 'error': 'No nodes provided'}), 400

        metrics = graph_evaluator.evaluate_graph(nodes, edges)
        return jsonify({'success': True, **metrics})

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Multi-Model Benchmark Endpoints ─────────────────────────────────────────

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """
    POST /api/benchmark
    Body: {
      "question": str,
      "reasoning_method": str,   # e.g. "cot"
      "models": [{"model": str, "provider": str, "api_key": str}, ...],
      "max_tokens": int,
      "chars_per_line": int,
      "max_lines": int
    }
    Returns ranked list of models with RQI scores.
    """
    try:
        data = request.json or {}
        question        = data.get('question', '').strip()
        reasoning_method= data.get('reasoning_method', 'cot')
        models          = data.get('models', [])
        max_tokens      = int(data.get('max_tokens', config.general.max_tokens))
        chars_per_line  = int(data.get('chars_per_line', config.general.chars_per_line))
        max_lines       = int(data.get('max_lines', config.general.max_lines))

        if not question:
            return jsonify({'success': False, 'error': 'question is required'}), 400
        if not models:
            return jsonify({'success': False, 'error': 'models list is required'}), 400

        viz_config = VisualizationConfig(
            max_chars_per_line=chars_per_line,
            max_lines=max_lines
        )

        run_results = []

        for model_entry in models:
            model_name = model_entry.get('model', '')
            provider   = model_entry.get('provider', 'openai')
            api_key    = model_entry.get('api_key', '')
            result_entry = {
                'model':    model_name,
                'provider': provider,
                'rqi':      None,
                's_coh':    None,
                's_eff':    None,
                's_struct': None,
                'raw_output': '',
                'error':    None,
            }

            try:
                # 1. Generate response
                api = create_api(provider, api_key, model_name)
                prompt_format = config.get_method_config(reasoning_method)
                pf = prompt_format.get('prompt_format') if prompt_format else None
                raw_response = api.generate_response(question, max_tokens=max_tokens,
                                                     prompt_format=pf)
                result_entry['raw_output'] = raw_response

                # 2. Parse graph
                parsed = None
                if reasoning_method == 'cot':
                    parsed = parse_cot_response(raw_response, question)
                elif reasoning_method == 'tot':
                    parsed = parse_tot_response(raw_response, question)
                elif reasoning_method == 'l2m':
                    parsed = parse_l2m_response(raw_response, question)
                elif reasoning_method == 'scr':
                    parsed = parse_scr_response(raw_response, question)
                elif reasoning_method == 'srf':
                    parsed = parse_selfrefine_response(raw_response, question)
                elif reasoning_method == 'bs':
                    parsed = parse_bs_response(raw_response, question)

                # 3. Extract nodes/edges
                if parsed:
                    g_nodes, g_edges = _extract_graph_data(parsed, reasoning_method)
                else:
                    g_nodes, g_edges = [], []

                # 4. Evaluate
                has_reasoning_nodes = any(node.get('id') != 'Q' for node in g_nodes)
                if g_nodes and has_reasoning_nodes:
                    metrics = graph_evaluator.evaluate_graph(g_nodes, g_edges)
                    result_entry.update({
                        'rqi':      metrics['rqi'],
                        's_coh':    metrics['s_coh'],
                        's_eff':    metrics['s_eff'],
                        's_struct': metrics['s_struct'],
                        'edge_scores': metrics.get('edge_scores', []),
                    })
                else:
                    result_entry['error'] = 'Graph extraction produced no reasoning or answer nodes'

            except Exception as model_err:
                logger.error(f"Benchmark error for {model_name}: {model_err}")
                result_entry['error'] = str(model_err)

            run_results.append(result_entry)
            from api_base import HuggingFaceLocalAPI
            if hasattr(HuggingFaceLocalAPI, 'free_memory'):
                HuggingFaceLocalAPI.free_memory()

        # Sort by RQI descending; models with errors go last
        run_results.sort(key=lambda r: (r['rqi'] is None, -(r['rqi'] or 0)))

        # Persist
        benchmark_record = {
            'id':        str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'prompt':    question,
            'method':    reasoning_method,
            'results':   run_results,
        }
        history = _load_benchmark_results()
        history.insert(0, benchmark_record)  # newest first
        history = history[:100]              # keep last 100 runs
        _save_benchmark_results(history)

        return jsonify({
            'success': True,
            'benchmark_id': benchmark_record['id'],
            'timestamp':    benchmark_record['timestamp'],
            'results':      run_results,
        })

    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/benchmark/history', methods=['GET'])
def benchmark_history():
    """GET /api/benchmark/history — return past benchmark runs (newest first)."""
    try:
        limit = int(request.args.get('limit', 20))
        history = _load_benchmark_results()
        return jsonify({'success': True, 'history': history[:limit]})
    except Exception as e:
        logger.error(f"Error fetching benchmark history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)

def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    try:
        # Run the application
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False  # Disable debug mode in production
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")