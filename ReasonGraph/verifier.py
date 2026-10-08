import json
import os
import sys

# Import our independent modules
from math_checker import check_math

# We'll load the NLI model lazily to meet the optional requirement
_nli_model = None

def get_nli_model():
    """Lazily load the NLI cross-encoder."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to load NLI model: {e}", file=sys.stderr)
    return _nli_model

def detect_contradictions(nodes):
    """
    Detects contradictions between a node and its parents.
    Returns flags for contradictions.
    """
    model = get_nli_model()
    if model is None:
        return {}
        
    flags = {}
    # Build text lookup for parents
    text_lookup = {node['id']: node['text'] for node in nodes}
    
    for node in nodes:
        node_id = node['id']
        node_text = node['text']
        parent_ids = node.get('parent_ids', [])
        
        for parent_id in parent_ids:
            if parent_id not in text_lookup:
                continue
                
            parent_text = text_lookup[parent_id]
            
            try:
                import torch
                scores = model.predict([(parent_text, node_text)])
                probs = torch.nn.functional.softmax(torch.tensor(scores), dim=-1)[0].tolist()
                
                contradiction_prob = probs[0]
                
                # Threshold for flagging
                if contradiction_prob > 0.85:
                    if node_id not in flags:
                        flags[node_id] = []
                    flags[node_id].append({
                        "type": "contradiction",
                        "reason": f"Contradicts step {parent_id} (confidence {contradiction_prob:.2f})",
                        "source": "nli",
                        "confidence": float(contradiction_prob)
                    })
            except Exception as e:
                pass
                
    return flags

def verify_graph(nodes):
    """
    Runs all verification checks (Math and NLI) on the nodes.
    Returns a dictionary of flags: { node_id: [ flags ] }
    """
    all_flags = {}
    
    # 1. Run Math Checks
    for node in nodes:
        node_id = node['id']
        text = node['text']
        
        math_errors = check_math(text)
        if math_errors:
            if node_id not in all_flags:
                all_flags[node_id] = []
            all_flags[node_id].extend(math_errors)
            
    # 2. Run NLI Checks (optional, lazy load)
    nli_flags = detect_contradictions(nodes)
    for node_id, flags in nli_flags.items():
        if node_id not in all_flags:
            all_flags[node_id] = []
        all_flags[node_id].extend(flags)
        
    return all_flags

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            sample_nodes = json.load(f)
        result = verify_graph(sample_nodes)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python verifier.py sample_graph.json")
