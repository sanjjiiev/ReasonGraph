import json
from verifier import verify_graph

def run_evaluation(cases_file):
    with open(cases_file, 'r') as f:
        cases = json.load(f)
        
    metrics = {
        'math_caught': 0,
        'math_missed': 0,
        'nli_caught': 0,
        'nli_missed': 0,
        'false_alarms': 0,
        'clean_cases': 0,
        'math_cases': 0,
        'nli_cases': 0
    }
    
    for case in cases:
        flags = verify_graph(case['nodes'])
        expected = set(case['expected_flags'])
        actual = set(flags.keys())
        
        is_clean = case['id'].startswith('clean-')
        is_math_err = case['id'].startswith('math-err-')
        is_nli_err = case['id'].startswith('nli-err-')
        
        if is_clean:
            metrics['clean_cases'] += 1
            if len(actual) > 0:
                metrics['false_alarms'] += 1
        elif is_math_err:
            metrics['math_cases'] += 1
            if expected.issubset(actual):
                metrics['math_caught'] += 1
            else:
                metrics['math_missed'] += 1
        elif is_nli_err:
            metrics['nli_cases'] += 1
            if expected.issubset(actual):
                metrics['nli_caught'] += 1
            else:
                metrics['nli_missed'] += 1
                
    print(f"Evaluation Results (N={len(cases)})")
    print("-" * 40)
    print(f"Math: caught {metrics['math_caught']}/{metrics['math_cases']} errors")
    print(f"NLI: caught {metrics['nli_caught']}/{metrics['nli_cases']} contradictions")
    print(f"False alarms: {metrics['false_alarms']} in {metrics['clean_cases']} clean cases")
    print("-" * 40)
    print("Model: cross-encoder/nli-deberta-v3-large | Threshold: 0.85")

if __name__ == "__main__":
    run_evaluation('eval_cases.json')
