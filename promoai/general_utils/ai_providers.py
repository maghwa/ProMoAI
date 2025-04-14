from enum import Enum

class AIProviders(Enum):
    OLLAMA = "Ollama"
    TOGETHER = "Together"

# Ollama models
OLLAMA_MODELS = {
    "gemma3:4b": "Google Gemma 4B",
    "deepseek-r1:latest": "DeepSeek R1",
    "deepcoder": "DeepCoder"
}

# Together models
TOGETHER_MODELS = {
    "meta-llama/Llama-3-70B-Instruct": "Llama 3 (70B)",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "togethercomputer/StripedHyena-Nous-7B": "Striped Hyena 7B"
}

AI_MODEL_DEFAULTS = {
    AIProviders.OLLAMA.value: 'gemma3:4b',
    AIProviders.TOGETHER.value: 'meta-llama/Llama-3-70B-Instruct'
}

DEFAULT_AI_PROVIDER = AIProviders.OLLAMA.value

AI_HELP_DEFAULTS = {
    AIProviders.OLLAMA.value: "Select an Ollama model that you have pulled locally (no API key required)",
    AIProviders.TOGETHER.value: "Select a Together AI model (requires API key from together.ai)"
}

MAIN_HELP = "Choose between local Ollama models or cloud-based Together AI models."