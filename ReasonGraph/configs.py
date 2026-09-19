from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_keys_from_file(file_path: str = "api_keys.json") -> Dict[str, str]:
    """Load API keys from a JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"API keys file {file_path} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading API keys from {file_path}: {str(e)}")
        return {}

@dataclass
class GeneralConfig:
    """General configuration parameters that are method-independent"""
    available_models: List[str] = field(default_factory=lambda: [
        # Anthropic Models
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        # OpenAI Models
        "gpt-4",
        "gpt-4-turbo", 
        "gpt-4-turbo-preview",
        "chatgpt-4o-latest",
        #"gpt-4o-mini",
        "gpt-3.5-turbo",
        # Gemini Models
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro-exp-02-05",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-2.0-flash-thinking-exp", 
        # Together AI Models
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        #"meta-llama/Llama-3.2-3B-Instruct-Turbo", 
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", 
        #"meta-llama/Meta-Llama-3.1-70B-Instruct-Reference", 
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", 
        #"meta-llama/Meta-Llama-3.1-8B-Instruct-Reference", 
        #"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Llama-3-70b-chat-hf", 
        "meta-llama/Meta-Llama-3-70B-Instruct", 
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo", 
        "meta-llama/Meta-Llama-3-70B-Instruct-Lite", 
        #"meta-llama/Llama-3-8b-chat-hf", 
        #"meta-llama/Meta-Llama-3-8B-Instruct", 
        #"meta-llama/Meta-Llama-3-8B-Instruct-Turbo", 
        #"meta-llama/Meta-Llama-3-8B-Instruct-Lite", 
        "deepseek-ai/DeepSeek-V3",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        #"microsoft/WizardLM-2-8x22B",
        #"databricks/dbrx-instruct",
        #"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        # DeepSeek Models
        "deepseek-chat",
        "deepseek-reasoner",
        # Qwen Models
        "qwen-max",
        "qwen-max-latest",
        "qwen-max-2025-01-25",
        "qwen-plus",
        "qwen-plus-latest",
        "qwen-plus-2025-01-25",
        "qwen-turbo",
        "qwen-turbo-latest",
        "qwen-turbo-2024-11-01",
        "qwq-plus",
        #"qwen2.5-14b-instruct-1m",
        #"qwen2.5-7b-instruct-1m",
        "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct",
        #"qwen2.5-14b-instruct",
        #"qwen2.5-7b-instruct",
        # Grok Models
        "grok-2",
        "grok-2-latest",
        # ── Local Models (Ollama) ─────────────────────────────────────────────
        # Requires Ollama running at http://localhost:11434
        # Pull models with: ollama pull <model>
        "llama3:8b",
        "llama3:70b",
        "llama3.1:8b",
        "llama3.1:70b",
        "mistral:7b",
        "mistral-nemo:12b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "qwen2.5:32b",
        "gemma3:9b",
        "phi4:14b",
        "deepseek-r1:8b",
        "deepseek-r1:14b",
        # Small local models (~1B-2B parameters)
        "gemma2:2b",
        "qwen2:1.5b",
        "qwen2.5:1.5b",
        "qwen3:1.7b",
        "deepseek-r1:1.5b",
        "smollm2:1.7b",
        "stablelm2:1.6b",
        "tinyllama:1.1b",
        # ── Local Models (HuggingFace direct load) ───────────────────────────
        # Requires: pip install transformers torch accelerate
        # Pass HF token as API key for gated models
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-4",
        # Small local models (~1B-2B parameters)
        "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3-1.7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "stabilityai/stablelm-2-zephyr-1_6b",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ])
    model_providers: Dict[str, str] = field(default_factory=lambda: {
        "claude-3-7-sonnet-20250219": "anthropic",
        "claude-3-5-sonnet-20241022": "anthropic",
        "claude-3-5-haiku-20241022": "anthropic",
        "claude-3-haiku-20240307": "anthropic",
        "claude-3-sonnet-20240229": "anthropic",
        "claude-3-opus-20240229": "anthropic",
        "gpt-4": "openai",
        "gpt-4-turbo": "openai", 
        "gpt-4-turbo-preview": "openai",
        "chatgpt-4o-latest": "openai",
        #"gpt-4o-mini": "openai",
        "gpt-3.5-turbo": "openai",
        "gemini-2.0-flash": "google",
        "gemini-2.0-flash-lite": "google",
        "gemini-2.0-pro-exp-02-05": "google",
        "gemini-1.5-flash": "google",
        "gemini-1.5-flash-8b": "google",
        "gemini-1.5-pro": "google",
        "gemini-2.0-flash-thinking-exp": "google", 
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
        #"meta-llama/Llama-3.2-3B-Instruct-Turbo": "together", 
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro": "together",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "together", 
        #"meta-llama/Meta-Llama-3.1-70B-Instruct-Reference": "together", 
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "together", 
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "together", 
        #"meta-llama/Meta-Llama-3.1-8B-Instruct-Reference": "together", 
        #"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
        "meta-llama/Llama-3-70b-chat-hf": "together", 
        "meta-llama/Meta-Llama-3-70B-Instruct": "together", 
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": "together", 
        "meta-llama/Meta-Llama-3-70B-Instruct-Lite": "together", 
        #"meta-llama/Llama-3-8b-chat-hf": "together", 
        #"meta-llama/Meta-Llama-3-8B-Instruct": "together", 
        #"meta-llama/Meta-Llama-3-8B-Instruct-Turbo": "together", 
        #"meta-llama/Meta-Llama-3-8B-Instruct-Lite": "together", 
        "deepseek-ai/DeepSeek-V3": "together",
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "together",
        "Qwen/Qwen2.5-72B-Instruct-Turbo": "together",
        #"microsoft/WizardLM-2-8x22B": "together",
        #"databricks/dbrx-instruct": "together",
        #"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "together",
        "deepseek-chat": "deepseek",
        "deepseek-reasoner": "deepseek",
        "qwen-max": "qwen",
        "qwen-max-latest": "qwen",
        "qwen-max-2025-01-25": "qwen",
        "qwen-plus": "qwen",
        "qwen-plus-latest": "qwen",
        "qwen-plus-2025-01-25": "qwen",
        "qwen-turbo": "qwen",
        "qwen-turbo-latest": "qwen",
        "qwen-turbo-2024-11-01": "qwen",
        "qwq-plus": "qwen",
        #"qwen2.5-14b-instruct-1m": "qwen",
        #"qwen2.5-7b-instruct-1m": "qwen",
        "qwen2.5-72b-instruct": "qwen",
        "qwen2.5-32b-instruct": "qwen",
        #"qwen2.5-14b-instruct": "qwen",
        #"qwen2.5-7b-instruct": "qwen",
        "grok-2": "grok",
        "grok-2-latest": "grok",
        # ── Local Models (Ollama) ─────────────────────────────────────────────
        "llama3:8b": "ollama",
        "llama3:70b": "ollama",
        "llama3.1:8b": "ollama",
        "llama3.1:70b": "ollama",
        "mistral:7b": "ollama",
        "mistral-nemo:12b": "ollama",
        "qwen2.5:7b": "ollama",
        "qwen2.5:14b": "ollama",
        "qwen2.5:32b": "ollama",
        "gemma3:9b": "ollama",
        "phi4:14b": "ollama",
        "deepseek-r1:8b": "ollama",
        "deepseek-r1:14b": "ollama",
        # Small local models (~1B-2B parameters)
        "gemma2:2b": "ollama",
        "qwen2:1.5b": "ollama",
        "qwen2.5:1.5b": "ollama",
        "qwen3:1.7b": "ollama",
        "deepseek-r1:1.5b": "ollama",
        "smollm2:1.7b": "ollama",
        "stablelm2:1.6b": "ollama",
        "tinyllama:1.1b": "ollama",
        # ── Local Models (HuggingFace direct load) ───────────────────────────
        "Qwen/Qwen2.5-7B-Instruct": "huggingface",
        "Qwen/Qwen2.5-14B-Instruct": "huggingface",
        "mistralai/Mistral-7B-Instruct-v0.3": "huggingface",
        "microsoft/Phi-4": "huggingface",
        # Small local models (~1B-2B parameters)
        "google/gemma-2-2b-it": "huggingface",
        "Qwen/Qwen2.5-1.5B-Instruct": "huggingface",
        "Qwen/Qwen3-1.7B": "huggingface",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "huggingface",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct": "huggingface",
        "stabilityai/stablelm-2-zephyr-1_6b": "huggingface",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "huggingface",
    })
    providers: List[str] = field(default_factory=lambda: [
        "anthropic", "openai", "google", "together", "deepseek", "qwen", "grok",
        "ollama", "huggingface",
    ])
    max_tokens: int = 2048
    chars_per_line: int = 40
    max_lines: int = 8
    
    def __post_init__(self):
        """Load API keys after initialization"""
        self.provider_api_keys = load_api_keys_from_file()

    def get_default_api_key(self, provider: str) -> str:
        """Get default API key for specific provider"""
        return self.provider_api_keys.get(provider, "")

@dataclass
class PlainTextConfig:
    """Configuration specific to Plain Text method (no visualization)"""
    name: str = "Plain Text"
    prompt_format: str = '''{question}'''
    example_question: str = "Who are you?"

@dataclass
class ChainOfThoughtsConfig:
    """Configuration specific to Chain of Thoughts method"""
    name: str = "Chain of Thoughts"
    prompt_format: str = '''Please answer the question using the following format by Chain-of-Thoughts, with each step clearly marked:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Final answer]
</answer>'''
    example_question: str = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

@dataclass
class TreeOfThoughtsConfig:
    """Configuration specific to Tree of Thoughts method"""
    name: str = "Tree of Thoughts"
    prompt_format: str = '''Please answer the question using Tree of Thoughts reasoning. Consider multiple possible approaches and explore their consequences. Feel free to create as many branches and sub-branches as needed for thorough exploration. Use the following format:

Question: {question}

Let's explore different paths of reasoning:
<node id="root">
[Initial analysis of the problem]
</node>

[Add main approaches with unique IDs (approach1, approach2, etc.)]
<node id="approach1" parent="root">
[First main approach to solve the problem]
</node>

[For each approach, add as many sub-branches as needed using parent references]
<node id="approach1.1" parent="approach1">
[Exploration of a sub-path]
</node>

[Continue adding nodes and exploring paths as needed. You can create deeper levels by extending the ID pattern (e.g., approach1.1.1)]

<answer>
Based on exploring all paths:
- [Explain which path(s) led to the best solution and why]
- [State the final answer]
</answer>'''
    example_question: str = "Using the numbers 3, 3, 8, and 8, find a way to make exactly 24 using basic arithmetic operations (addition, subtraction, multiplication, division). Each number must be used exactly once, and you can use parentheses to control the order of operations."

@dataclass
class LeastToMostConfig:
    """Configuration specific to Least-to-Most method"""
    name: str = "Least to Most"
    prompt_format: str = '''Please solve this question using the Least-to-Most approach. First break down the complex question into simpler sub-questions, then solve them in order from simplest to most complex.

Question: {question}

Let's solve this step by step:
<step number="1">
<question>[First sub-question - should be the simplest]</question>
<reasoning>[Reasoning process for this sub-question]</reasoning>
<answer>[Answer to this sub-question]</answer>
</step>
... (add more steps as needed)
<final_answer>
[Final answer that combines the insights from all steps]
</final_answer>'''
    example_question: str = "How to create a personal website?"

@dataclass
class SelfRefineConfig:
    """Configuration specific to Self-Refine method"""
    name: str = "Self-Refine"
    prompt_format: str = '''Please solve this question step by step, then check your work and revise any mistakes. Use the following format:

Question: {question}

Let's solve this step by step:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Initial answer]
</answer>

Now, let's check our work:
<revision_check>
[Examine each step for errors or improvements]
</revision_check>

[If any revisions are needed, add revised steps:]
<revised_step number="[new_step_number]" revises="[original_step_number]">
[Corrected reasoning]
</revised_step>
... (add more revised steps if needed)

[If the answer changes, add the revised answer:]
<revised_answer>
[Updated final answer]
</revised_answer>'''
    example_question: str = "Write a one sentence fiction and then improve it after refine."

@dataclass
class SelfConsistencyConfig:
    """Configuration specific to Self-consistency method"""
    name: str = "Self-consistency"
    prompt_format: str = '''Please solve the question using multiple independent reasoning paths. Generate 3 different Chain-of-Thought solutions and provide the final answer based on majority voting.

Question: {question}

Path 1:
<step number="1">
[First step of reasoning]
</step>
... (add more steps as needed)
<answer>
[Path 1's answer]
</answer>

Path 2:
... (repeat the same format for all 3 paths)

Note: Each path should be independent and may arrive at different answers. The final answer will be determined by majority voting.'''
    example_question: str = "How many r are there in strawberrrrrrrrry?"

@dataclass
class BeamSearchConfig:
    """Configuration specific to Beam Search method"""
    name: str = "Beam Search"
    prompt_format: str = '''Please solve this question using Beam Search reasoning. For each step:
1. Explore multiple paths fully regardless of intermediate scores
2. Assign a score between 0 and 1 to each node based on how promising that step is
3. Calculate path_score for each result by summing scores along the path from root to result
4. The final choice will be based on the highest cumulative path score

Question: {question}

<node id="root" score="[score]">
[Initial analysis - Break down the key aspects of the problem]
</node>

# First approach branch
<node id="approach1" parent="root" score="[score]">
[First approach - Outline the general strategy]
</node>

<node id="impl1.1" parent="approach1" score="[score]">
[Implementation 1.1 - Detail the specific steps and methods]
</node>

<node id="result1.1" parent="impl1.1" score="[score]" path_score="[sum of scores from root to here]">
[Result 1.1 - Describe concrete outcome and effectiveness]
</node>

<node id="impl1.2" parent="approach1" score="[score]">
[Implementation 1.2 - Detail alternative steps and methods]
</node>

<node id="result1.2" parent="impl1.2" score="[score]" path_score="[sum of scores from root to here]">
[Result 1.2 - Describe concrete outcome and effectiveness]
</node>

# Second approach branch
<node id="approach2" parent="root" score="[score]">
[Second approach - Outline an alternative general strategy]
</node>

<node id="impl2.1" parent="approach2" score="[score]">
[Implementation 2.1 - Detail the specific steps and methods]
</node>

<node id="result2.1" parent="impl2.1" score="[score]" path_score="[sum of scores from root to here]">
[Result 2.1 - Describe concrete outcome and effectiveness]
</node>

<node id="impl2.2" parent="approach2" score="[score]">
[Implementation 2.2 - Detail alternative steps and methods]
</node>

<node id="result2.2" parent="impl2.2" score="[score]" path_score="[sum of scores from root to here]">
[Result 2.2 - Describe concrete outcome and effectiveness]
</node>

<answer>
Best path (path_score: [highest_path_score]):
[Identify the path with the highest cumulative score]
[Explain why this path is most effective]
[Provide the final synthesized solution]
</answer>'''
    example_question: str = "Give me two suggestions for transitioning from a journalist to a book editor?"

class ReasoningConfig:
    """Main configuration class that manages both general and method-specific configs"""
    def __init__(self):
        self.general = GeneralConfig()
        self.methods = {
            "cot": ChainOfThoughtsConfig(),
            "tot": TreeOfThoughtsConfig(),
            "scr": SelfConsistencyConfig(),
            "srf": SelfRefineConfig(),
            "l2m": LeastToMostConfig(),
            "bs": BeamSearchConfig(),
            "plain": PlainTextConfig(),
        }
    
    def get_method_config(self, method_id: str) -> Optional[dict]:
        """Get configuration for specific method"""
        method = self.methods.get(method_id)
        if method:
            return {
                "name": method.name,
                "prompt_format": method.prompt_format,
                "example_question": method.example_question
            }
        return None

    def get_initial_values(self) -> dict:
        """Get initial values for UI"""
        return {
            "general": {
                "available_models": self.general.available_models,
                "model_providers": self.general.model_providers,
                "providers": self.general.providers,
                "max_tokens": self.general.max_tokens,
                "default_api_key": self.general.get_default_api_key(self.general.providers[0]),
                "visualization": {
                    "chars_per_line": self.general.chars_per_line,
                    "max_lines": self.general.max_lines
                }
            },
            "methods": {
                method_id: {
                    "name": config.name,
                    "prompt_format": config.prompt_format,
                    "example_question": config.example_question
                }
                for method_id, config in self.methods.items()
            }
        }
    
    def add_method(self, method_id: str, config: Any) -> None:
        """Add a new reasoning method configuration"""
        if method_id not in self.methods:
            self.methods[method_id] = config
        else:
            raise ValueError(f"Method {method_id} already exists")

# Create global config instance
config = ReasoningConfig()