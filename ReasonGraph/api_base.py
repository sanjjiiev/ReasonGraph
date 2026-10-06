"""
Base API module for handling different API providers.
This module provides a unified interface for interacting with various API providers
like Anthropic, OpenAI, Google Gemini, Together AI, as well as local providers
(Ollama/vLLM via OpenAI-compatible endpoint, and direct HuggingFace model loading).
"""

from abc import ABC, abstractmethod
import logging
import os
import requests
from openai import OpenAI
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "180"))
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response structure"""
    text: str
    raw_response: Any
    usage: Dict[str, int]
    model: str

class APIError(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} API Error: {message} (Status: {status_code})")

class BaseAPI(ABC):
    """Abstract base class for API interactions"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider_name = "base"  # Override in subclasses
        
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the API"""
        pass

    def _format_prompt(self, question: str, prompt_format: Optional[str] = None) -> str:
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

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """Standardized error handling"""
        error_msg = f"{self.provider_name} API error in {context}: {str(error)}"
        logger.error(error_msg)
        raise APIError(str(error), self.provider_name)

class AnthropicAPI(BaseAPI):
    """Class to handle interactions with the Anthropic API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key, model)
        self.provider_name = "Anthropic"
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Anthropic API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "max_tokens": max_tokens
            }
            
            logger.info(f"Sending request to Anthropic API with model {self.model}")
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            self._handle_error(e, "request")
        except (KeyError, IndexError) as e:
            self._handle_error(e, "response parsing")
        except Exception as e:
            self._handle_error(e, "unexpected")

class OpenAIAPI(BaseAPI):
    """Class to handle interactions with the OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        super().__init__(api_key, model)
        self.provider_name = "OpenAI"
        self.request_timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        self.max_retries = DEFAULT_MAX_RETRIES
        try:
            self.client = OpenAI(
                api_key=api_key,
                timeout=self.request_timeout_seconds,
                max_retries=self.max_retries,
            )
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the OpenAI API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(
                f"Sending request to OpenAI API with model {self.model} "
                f"(timeout={self.request_timeout_seconds}s, retries={self.max_retries})"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=max_tokens,
                timeout=self.request_timeout_seconds,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                raise APIError(
                    f"Request timed out after {self.request_timeout_seconds}s. Please try again or increase OPENAI_TIMEOUT_SECONDS.",
                    self.provider_name,
                )
            self._handle_error(e, "request or response processing")

class GeminiAPI(BaseAPI):
    """Class to handle interactions with the Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        super().__init__(api_key, model)
        self.provider_name = "Gemini"
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Gemini API"""
        try:
            from google.genai import types
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Gemini API with model {self.model}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=[formatted_prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            
            if not response.text:
                raise APIError("Empty response from Gemini API", self.provider_name)
                
            return response.text
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class TogetherAPI(BaseAPI):
    """Class to handle interactions with the Together AI API"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        super().__init__(api_key, model)
        self.provider_name = "Together"
        try:
            from together import Together
            self.client = Together(api_key=api_key)
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Together AI API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Together AI API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=max_tokens
            )
            
            # Robust response extraction
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            elif hasattr(response, 'text'):
                return response.text
            else:
                # If response doesn't match expected structures
                raise APIError("Unexpected response format from Together AI", self.provider_name)
            
        except Exception as e:
            self._handle_error(e, "request or response processing")
class DeepSeekAPI(BaseAPI):
    """Class to handle interactions with the DeepSeek API via OpenRouter"""
    
    def __init__(self, api_key: str, model: str = "deepseek/deepseek-chat"):
        super().__init__(api_key, model)
        self.provider_name = "DeepSeek"
        try:
            self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the DeepSeek API via OpenRouter"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            # Map standard model dropdown values to OpenRouter's required IDs
            model_id = self.model
            if model_id == "deepseek-chat":
                model_id = "deepseek/deepseek-chat"
            elif model_id == "deepseek-reasoner":
                model_id = "deepseek/deepseek-r1"
            
            logger.info(f"Sending request to DeepSeek (via OpenRouter) with model {model_id}")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=max_tokens
            )
            
            # Check if this is the reasoning model response
            if ("reasoner" in self.model or "r1" in model_id) and hasattr(response.choices[0].message, "reasoning_content"):
                reasoning = response.choices[0].message.reasoning_content
                answer = response.choices[0].message.content
                return f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}"
            else:
                # Regular model response
                return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")
class QwenAPI(BaseAPI):
    """Class to handle interactions with the Qwen API"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        super().__init__(api_key, model)
        self.provider_name = "Qwen"
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Qwen API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Qwen API with model {self.model}")
            
            # Check if this is the reasoning model (qwq-plus)
            if self.model == "qwq-plus":
                # For qwq-plus model, we need to use streaming
                reasoning_content = ""
                answer_content = ""
                is_answering = False
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    max_tokens=max_tokens,
                    stream=True  # qwq-plus only supports streaming output
                )
                
                for chunk in response:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    # Collect reasoning process
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                    # Collect answer content
                    elif hasattr(delta, 'content') and delta.content is not None:
                        answer_content += delta.content
                        is_answering = True
                
                # Return combined reasoning and answer
                return f"Reasoning:\n{reasoning_content}\n\nAnswer:\n{answer_content}"
            else:
                # Regular model response (non-streaming)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class GrokAPI(BaseAPI):
    """Class to handle interactions with the Grok API"""
    
    def __init__(self, api_key: str, model: str = "grok-2-latest"):
        super().__init__(api_key, model)
        self.provider_name = "Grok"
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(self, prompt: str, max_tokens: int = 1024, 
                         prompt_format: Optional[str] = None) -> str:
        """Generate a response using the Grok API"""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            
            logger.info(f"Sending request to Grok API with model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self._handle_error(e, "request or response processing")

class LocalOllamaAPI(BaseAPI):
    """
    Class to handle interactions with a local Ollama or vLLM server via its
    OpenAI-compatible REST endpoint.

    Ollama default base URL : http://localhost:11434/v1
    vLLM  default base URL  : http://localhost:8000/v1

    No real API key is required; pass an empty string or 'ollama'.
    """

    def __init__(
        self,
        api_key: str = "ollama",
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434/v1",
    ):
        # Use a dummy non-empty key so the OpenAI client doesn't complain
        super().__init__(api_key or "ollama", model)
        self.provider_name = "Ollama"
        self.base_url = base_url
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except Exception as e:
            self._handle_error(e, "initialization")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        prompt_format: Optional[str] = None,
    ) -> str:
        """Generate a response using the local Ollama / vLLM OpenAI-compatible API."""
        try:
            formatted_prompt = self._format_prompt(prompt, prompt_format)
            logger.info(
                f"Sending request to local Ollama/vLLM at {self.base_url} "
                f"with model {self.model}"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self._handle_error(e, "request or response processing")


def _detect_torch_device():
    """
    Auto-detect the best available torch device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    Works on Windows (CUDA), macOS M-series (MPS), and any CPU-only machine.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


class HuggingFaceLocalAPI(BaseAPI):
    """
    Class to load a Hugging Face model directly (no external API call).

    The model weights are loaded once and cached for the lifetime of the object.
    Uses bfloat16 on CUDA and float32 on MPS / CPU.

    Requires: transformers, torch, accelerate
    Example model IDs:
      - "meta-llama/Llama-3.1-8B-Instruct"   (requires HF token)
      - "mistralai/Mistral-7B-Instruct-v0.3"
      - "Qwen/Qwen2.5-7B-Instruct"
    """

    # Class-level cache: model_id -> (model, tokenizer)
    _loaded_models: Dict[str, Any] = {}

    def __init__(self, api_key: str = "", model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Parameters
        ----------
        api_key : str
            Optional Hugging Face Hub token (needed for gated models like Llama).
            Pass as the 'API key' in the UI.
        model : str
            HuggingFace Hub model identifier.
        """
        super().__init__(api_key, model)
        self.provider_name = "HuggingFace"
        self._ensure_model_loaded()

    
    @classmethod
    def free_memory(cls):
        cls._loaded_models.clear()
        import gc
        gc.collect()
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    def _ensure_model_loaded(self) -> None:
        """Lazy-load model + tokenizer, caching at class level."""
        if self.model in HuggingFaceLocalAPI._loaded_models:
            self._model, self._tokenizer = HuggingFaceLocalAPI._loaded_models[self.model]
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = _detect_torch_device()
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            hf_token = self.api_key if self.api_key and self.api_key != "huggingface" else None

            logger.info(
                f"Loading HuggingFace model '{self.model}' on device='{device}' "
                f"dtype={dtype} ..."
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                token=hf_token,
                trust_remote_code=True,
            )

            # device_map='auto' handles multi-GPU / CPU offloading automatically;
            # on MPS we pass the device string directly because device_map is
            # not yet fully supported there.
            if device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=dtype,
                    token=hf_token,
                    trust_remote_code=True,
                ).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    device_map="auto",
                    torch_dtype=dtype,
                    token=hf_token,
                    trust_remote_code=True,
                )

            HuggingFaceLocalAPI._loaded_models[self.model] = (model, tokenizer)
            self._model, self._tokenizer = model, tokenizer
            logger.info(f"Model '{self.model}' loaded successfully.")
        except ImportError as e:
            raise APIError(
                "transformers and torch are required for HuggingFaceLocalAPI. "
                "Install with: pip install transformers torch accelerate",
                self.provider_name,
            ) from e
        except Exception as e:
            self._handle_error(e, "model loading")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        prompt_format: Optional[str] = None,
    ) -> str:
        """Generate a response using the locally loaded HuggingFace model."""
        try:
            import torch

            formatted_prompt = self._format_prompt(prompt, prompt_format)

            # Build chat messages and apply the model's chat template
            messages = [{"role": "user", "content": formatted_prompt}]
            if hasattr(self._tokenizer, "apply_chat_template"):
                input_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_text = formatted_prompt

            device = _detect_torch_device()
            inputs = self._tokenizer(input_text, return_tensors="pt").to(device)

            logger.info(
                f"Generating with HuggingFace model '{self.model}' "
                f"(max_new_tokens={max_tokens})"
            )
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        except Exception as e:
            self._handle_error(e, "request or response processing")


class APIFactory:

    """Factory class for creating API instances"""
    
    _providers = {
        "anthropic": {
            "class": AnthropicAPI,
            "default_model": "claude-3-7-sonnet-20250219"
        },
        "openai": {
            "class": OpenAIAPI,
            "default_model": "gpt-4-turbo-preview"
        },
        "google": {
            "class": GeminiAPI,
            "default_model": "gemini-2.0-flash"
        },
        "together": {
            "class": TogetherAPI,
            "default_model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        },
        "deepseek": {
            "class": DeepSeekAPI,
            "default_model": "deepseek/deepseek-chat"
        },
        "qwen": {
            "class": QwenAPI,
            "default_model": "qwen-plus"
        },
        "grok": {
            "class": GrokAPI,
            "default_model": "grok-2-latest"
        },
        # ── Local providers (no cloud key required) ──────────────────────────
        "ollama": {
            "class": LocalOllamaAPI,
            "default_model": "llama3:8b"
        },
        "huggingface": {
            "class": HuggingFaceLocalAPI,
            "default_model": "Qwen/Qwen2.5-7B-Instruct"
        },
    }
    
    @classmethod
    def supported_providers(cls) -> List[str]:
        """Get list of supported providers"""
        return list(cls._providers.keys())
    
    @classmethod
    def create_api(cls, provider: str, api_key: str, model: Optional[str] = None) -> BaseAPI:
        """Factory method to create appropriate API instance"""
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported providers are: {', '.join(cls.supported_providers())}")
        
        provider_info = cls._providers[provider]
        api_class = provider_info["class"]
        model = model or provider_info["default_model"]

        # Local providers don't need a real API key
        if provider == "ollama" and not api_key:
            api_key = "ollama"
        
        logger.info(f"Creating API instance for provider: {provider}, model: {model}")

        return api_class(api_key=api_key, model=model)

def create_api(provider: str, api_key: str, model: Optional[str] = None) -> BaseAPI:
    """Convenience function to create API instance"""
    return APIFactory.create_api(provider, api_key, model)

# Example usage:
if __name__ == "__main__":
    # Example with Anthropic
    anthropic_api = create_api("anthropic", "your-api-key")
    
    # Example with OpenAI
    openai_api = create_api("openai", "your-api-key", "gpt-4")
    
    # Example with Gemini
    gemini_api = create_api("gemini", "your-api-key", "gemini-2.0-flash")
    
    # Example with Together AI
    together_api = create_api("together", "your-api-key")
    
    # Get supported providers
    providers = APIFactory.supported_providers()
    print(f"Supported providers: {providers}")