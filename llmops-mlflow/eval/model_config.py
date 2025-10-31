from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    url: str
    temperature: float = 0.7

# Simple Ollama configuration
OLLAMA_MODEL = ModelConfig(
    name="llama3.2",
    url="http://localhost:11434/v1",
    temperature=0.7
)

# If you want to compare with cloud model later
CLOUD_MODEL = ModelConfig(
    name="deepseek",
    url="http://your-server:8000/v1",
    temperature=0.0
)

# Currently using only Ollama model
MODELS_TO_EVALUATE = [OLLAMA_MODEL]