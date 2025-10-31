import time
from typing import Any, Callable
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def _norm(s: str) -> str:
    """Normalize text for comparison."""
    return "".join(c.lower() for c in str(s) if c.isalnum())

class MetricsCalculator:
    def __init__(self):
        self.tokenizer = None
        self.semantic_model = None
    
    def _get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return self.tokenizer
    
    def _get_semantic_model(self):
        if self.semantic_model is None:
            self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return self.semantic_model

    def exact_match(self, row: pd.Series) -> float:
        """Calculate exact match score."""
        if isinstance(row["answer"], list):
            return float(any(_norm(row["prediction"]) == _norm(ans) for ans in row["answer"]))
        return float(_norm(row["prediction"]) == _norm(row["answer"]))

    def contains_answer(self, row: pd.Series) -> float:
        """Calculate contains score."""
        if isinstance(row["answer"], list):
            return float(any(_norm(ans) in _norm(row["prediction"]) for ans in row["answer"]))
        return float(_norm(row["answer"]) in _norm(row["prediction"]))

    def token_ratio(self, row: pd.Series) -> float:
        """Calculate approximate token length ratio using whitespace tokenization."""
        def count_tokens(text: str) -> int:
            # Simple approximation: split on whitespace and punctuation
            return len(str(text).split())
        
        pred_tokens = count_tokens(row["prediction"])
        if isinstance(row["answer"], list):
            ref_tokens = min(count_tokens(ans) for ans in row["answer"])
        else:
            ref_tokens = count_tokens(row["answer"])
        ratio = min(pred_tokens / max(1, ref_tokens), 3.0)  # Cap at 3.0
        return ratio

    def semantic_similarity(self, row: pd.Series) -> float:
        """Calculate semantic similarity score."""
        model = self._get_semantic_model()
        pred_emb = model.encode(row["prediction"], convert_to_tensor=True)
        if isinstance(row["answer"], list):
            ref_embs = [model.encode(ans, convert_to_tensor=True) for ans in row["answer"]]
            scores = [float(pred_emb @ ref_emb.T) for ref_emb in ref_embs]
            return max(scores)
        ref_emb = model.encode(row["answer"], convert_to_tensor=True)
        return float(pred_emb @ ref_emb.T)

    def measure_latency(self, client: OpenAI, model_name: str, temperature: float) -> Callable:
        """Create a latency measurement function."""
        def _measure(row: pd.Series) -> float:
            start_time = time.time()
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": row["question"]}],
                temperature=temperature,
            )
            return time.time() - start_time
        return _measure

    def get_all_metrics(self) -> dict[str, Callable]:
        """Get all available metrics functions."""
        return {
            "exact_match": self.exact_match,
            "contains": self.contains_answer,
            "token_ratio": self.token_ratio,
            "semantic_similarity": self.semantic_similarity
        }

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for question-answering. Analyze the response based on:
1. Accuracy (0-0.4): How factually correct is the response?
2. Completeness (0-0.3): Does it fully answer all parts of the question?
3. Clarity (0-0.3): Is the response clear and well-structured?

Sum these components for a final score between 0 and 1. Return only the final score as a decimal number."""

def judge_llm(client: OpenAI, model_name: str) -> Callable:
    """Create an LLM-based judge function."""
    def _judge(row: pd.Series) -> float:
        prompt = f"QUESTION: {row['question']}\nPREDICTION: {row['prediction']}\nREFERENCE: {row['answer']}\nScore:"
        r = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        txt = r.choices[0].message.content.strip()
        try:
            val = float(txt)
            return max(0.0, min(1.0, val))
        except:
            import re
            m = re.search(r"([01](?:\.\d+)?)", txt)
            return float(m.group(1)) if m else 0.0
    
    return _judge