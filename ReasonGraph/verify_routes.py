from flask import Blueprint, request, jsonify
from verifier import verify_graph

verify_bp = Blueprint('verify_bp', __name__)

@verify_bp.route('/api/verify', methods=['POST'])
def verify_api():
    """
    POST /api/verify
    Body: {"nodes": [{"id": "...", "text": "...", "parent_ids": ["..."]}]}
    Returns JSON of flags per node_id.
    """
    try:
        data = request.json or {}
        nodes = data.get('nodes', [])
        
        if not nodes:
            return jsonify({'success': False, 'error': 'No nodes provided'}), 400
            
        flags = verify_graph(nodes)
        
        return jsonify({
            'success': True,
            'flags': flags
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
