import re
import operator

def check_math(text):
    """
    Scans text for arithmetic equations and verifies their correctness.
    Returns a list of error dictionaries if math is wrong.
    """
    errors = []
    
    # Regex to find simple equations like: "48 / 2 = 25" or "10 + 5 = 15"
    # Matches: number, optional space, operator, optional space, number, optional space, =, optional space, number
    pattern = r'(-?\d+(?:\.\d+)?)\s*([\+\-\*/x])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)'
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        'x': operator.mul,
        '/': operator.truediv
    }
    
    for match in matches:
        left_str, op_str, right_str, eq_str = match.groups()
        
        try:
            left = float(left_str)
            right = float(right_str)
            reported_result = float(eq_str)
            op_str = op_str.lower()
            
            # Avoid division by zero
            if op_str == '/' and right == 0:
                continue
                
            expected_result = ops[op_str](left, right)
            
            # Allow minor floating point inaccuracies
            if abs(expected_result - reported_result) > 1e-4:
                # Format expected cleanly (remove .0 if integer)
                if expected_result.is_integer():
                    expected_str = str(int(expected_result))
                else:
                    expected_str = f"{expected_result:.4g}"
                    
                errors.append({
                    "type": "arithmetic",
                    "reason": f"{left_str} {op_str} {right_str} = {expected_str}, not {eq_str}",
                    "source": "math_checker",
                    "confidence": 1.0
                })
        except ValueError:
            continue
            
    return errors
