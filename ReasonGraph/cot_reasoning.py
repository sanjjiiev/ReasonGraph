import re
import requests
import textwrap
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CoTStep:
    """Data class representing a single CoT step"""
    number: int
    content: str

@dataclass
class CoTResponse:
    """Data class representing a complete CoT response"""
    question: str
    steps: List[CoTStep]
    answer: Optional[str] = None

@dataclass
class VisualizationConfig:
    """Configuration for CoT visualization"""
    max_chars_per_line: int = 40
    max_lines: int = 4
    truncation_suffix: str = "..."

class AnthropicAPI:
    """Class to handle interactions with the Anthropic API"""
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def generate_response(self, prompt: str, max_tokens: int = 1024, prompt_format: str = None) -> str:
        """Generate a response using the Anthropic API"""
        formatted_prompt = self._format_prompt(prompt, prompt_format) if prompt_format else prompt
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def _format_prompt(self, question: str, prompt_format: str = None) -> str:
        """Format the prompt using custom format if provided"""
        if prompt_format:
            return prompt_format.format(question=question)
        
        # Default format if none provided
        return f"""Please answer the question using the following format, with each step clearly marked:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
<step number="2">
[Second step of reasoning]
</step>
<step number="3">
[Third step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Final answer]
</answer>

Note:
1. Each step must be wrapped in XML tags <step>
2. Each step must have a number attribute
3. The final answer must be wrapped in <answer> tags
"""

def wrap_text(text: str, config: VisualizationConfig) -> str:
    """Wrap text to fit within box constraints"""
    text = text.replace('\n', ' ').replace('"', "'")
    wrapped_lines = textwrap.wrap(text, width=config.max_chars_per_line)
    
    if len(wrapped_lines) > config.max_lines:
        # Option 1: Simply truncate and add ellipsis to the last line
        wrapped_lines = wrapped_lines[:config.max_lines]
        wrapped_lines[-1] = wrapped_lines[-1][:config.max_chars_per_line-3] + "..."
        
        # Option 2 (alternative): Include part of the next line to show continuity
        # original_next_line = wrapped_lines[config.max_lines] if len(wrapped_lines) > config.max_lines else ""
        # wrapped_lines = wrapped_lines[:config.max_lines-1]
        # wrapped_lines.append(original_next_line[:config.max_chars_per_line-3] + "...")
    
    return "<br>".join(wrapped_lines)

def parse_cot_response(response_text: str, question: str) -> CoTResponse:
    """
    Parse CoT response text to extract steps and final answer.
    
    Args:
        response_text: The raw response from the API
        question: The original question
    
    Returns:
        CoTResponse object containing question, steps, and answer
    """
    # Extract all steps
    step_pattern = r'<step number="(\d+)">\s*(.*?)\s*</step>'
    steps = []
    for match in re.finditer(step_pattern, response_text, re.DOTALL):
        number = int(match.group(1))
        content = match.group(2).strip()
        steps.append(CoTStep(number=number, content=content))

    plain_steps_parsed = False
    if not steps:
        plain_step_matches = list(re.finditer(
            r'^\s*(?:step\s*)?(\d+)\s*[.):]\s*(.*)$',
            response_text,
            re.IGNORECASE | re.MULTILINE,
        ))
        plain_steps_parsed = bool(plain_step_matches)
        for index, match in enumerate(plain_step_matches):
            end = (
                plain_step_matches[index + 1].start()
                if index + 1 < len(plain_step_matches)
                else len(response_text)
            )
            answer_header = re.search(
                r'^\s*(?:final\s+)?answer\s*:',
                response_text[match.end():end],
                re.IGNORECASE | re.MULTILINE,
            )
            if answer_header:
                end = match.end() + answer_header.start()
            content = "\n".join(
                part.strip()
                for part in (match.group(2), response_text[match.end():end])
                if part.strip()
            )
            if content:
                steps.append(CoTStep(number=len(steps) + 1, content=content))
    
    # Extract answer
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
    if not answer_match:
        answer_match = re.search(
            r'^\s*(?:final\s+)?answer\s*:\s*(.+?)(?=\n\s*\n|\n\s*(?:step\s*)?\d+\s*[.):]|\Z)',
            response_text,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
    answer = answer_match.group(1).strip() if answer_match else None
    
    if not plain_steps_parsed:
        steps.sort(key=lambda x: x.number)
    
    return CoTResponse(question=question, steps=steps, answer=answer)

def create_mermaid_diagram(cot_response: CoTResponse, config: VisualizationConfig) -> str:
    """
    Convert CoT steps to Mermaid diagram with improved text wrapping.
    
    Args:
        cot_response: CoTResponse object containing the reasoning steps
        config: VisualizationConfig for text formatting
    
    Returns:
        Mermaid diagram markup as a string
    """
    diagram = ['<div class="mermaid">', 'graph TD']
    
    # Add question node
    question_content = wrap_text(cot_response.question, config)
    diagram.append(f'    Q["{question_content}"]')
    
    # Add steps with wrapped text and connect them
    if cot_response.steps:
        # Connect question to first step
        diagram.append(f'    Q --> S{cot_response.steps[0].number}')
        
        # Add all steps
        for i, step in enumerate(cot_response.steps):
            content = wrap_text(step.content, config)
            node_id = f'S{step.number}'
            diagram.append(f'    {node_id}["{content}"]')
            
            # Connect steps sequentially
            if i < len(cot_response.steps) - 1:
                next_id = f'S{cot_response.steps[i + 1].number}'
                diagram.append(f'    {node_id} --> {next_id}')
    
    # Add final answer node
    if cot_response.answer:
        answer = wrap_text(cot_response.answer, config)
        diagram.append(f'    A["{answer}"]')
        if cot_response.steps:
            diagram.append(f'    S{cot_response.steps[-1].number} --> A')
        else:
            diagram.append('    Q --> A')
    
    # Add styles for better visualization
    diagram.extend([
        '    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    class Q question;',
        '    class A answer;',
        '    linkStyle default stroke:#666,stroke-width:2px;'
    ])
    
    diagram.append('</div>')
    return '\n'.join(diagram)