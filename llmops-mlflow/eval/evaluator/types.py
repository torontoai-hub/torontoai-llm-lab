from dataclasses import dataclass
from typing import Optional, Dict, List, Any

@dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    temperature: float
    description: str
    environment: str  # 'local' or 'cloud'
    hardware: str  # e.g., 'M1 Pro', 'A100'
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None

@dataclass
class EvaluationQuestion:
    question: str
    answer: str | List[str]
    category: str
    complexity: str

@dataclass
class EvaluationResult:
    model_name: str
    environment: str
    hardware: str
    metrics: Dict[str, float]
    detailed_results: Any  # DataFrame or MLflow results
    run_id: Optional[str] = None