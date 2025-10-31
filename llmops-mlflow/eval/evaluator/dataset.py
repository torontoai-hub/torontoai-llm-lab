import pandas as pd
from typing import Dict, List
from .types import EvaluationQuestion

EVAL_CATEGORIES: Dict[str, List[EvaluationQuestion]] = {
    "arithmetic": [
        EvaluationQuestion("What is 13 + 29?", "42", "arithmetic", "simple"),
        EvaluationQuestion(
            "If you have 5 apples and eat 2, how many do you have left?",
            "3",
            "arithmetic",
            "word_problem"
        ),
        EvaluationQuestion("Calculate: 15% of 200", "30", "arithmetic", "percentage"),
    ],
    "knowledge": [
        EvaluationQuestion(
            "Capital of Canada (one word)?",
            "Ottawa",
            "knowledge",
            "simple"
        ),
        EvaluationQuestion(
            "What is the largest planet in our solar system?",
            "Jupiter",
            "knowledge",
            "simple"
        ),
        EvaluationQuestion(
            "Who wrote 'Romeo and Juliet'?",
            "William Shakespeare",
            "knowledge",
            "literature"
        ),
    ],
    "reasoning": [
        EvaluationQuestion(
            "Sort ascending: 9, 2, 11, 3.",
            "2, 3, 9, 11",
            "reasoning",
            "sorting"
        ),
        EvaluationQuestion(
            "If all mammals are warm-blooded, and whales are mammals, are whales warm-blooded?",
            "yes",
            "reasoning",
            "logic"
        ),
        EvaluationQuestion(
            "What comes next in the sequence: 2, 4, 8, 16, ?",
            "32",
            "reasoning",
            "pattern"
        ),
    ],
    "language": [
        EvaluationQuestion(
            "Translate to French: 'apple'",
            "pomme",
            "language",
            "translation"
        ),
        EvaluationQuestion(
            "What is the plural of 'child'?",
            "children",
            "language",
            "grammar"
        ),
        EvaluationQuestion(
            "Give a synonym for 'happy'",
            ["joyful", "glad", "delighted", "cheerful"],
            "language",
            "vocabulary"
        ),
    ],
    "open_ended": [
        EvaluationQuestion(
            "Explain how a bicycle works in one sentence.",
            "A bicycle works by converting the rider's pedaling energy through a chain-drive system to rotate wheels, creating forward motion.",
            "open_ended",
            "explanation"
        ),
        EvaluationQuestion(
            "What are three benefits of exercise?",
            "Exercise improves cardiovascular health, helps maintain healthy weight, and reduces stress.",
            "open_ended",
            "listing"
        ),
    ],
    "edge_cases": [
        EvaluationQuestion(
            "",
            "I cannot provide a response without a question.",
            "edge_cases",
            "empty"
        ),
        EvaluationQuestion(
            "?????",
            "I cannot understand your question. Please provide a clear question.",
            "edge_cases",
            "unclear"
        ),
    ],
}

def load_evaluation_dataset() -> pd.DataFrame:
    """Convert the evaluation questions into a pandas DataFrame."""
    records = []
    for category, questions in EVAL_CATEGORIES.items():
        records.extend([{
            'question': q.question,
            'answer': q.answer,
            'category': q.category,
            'complexity': q.complexity
        } for q in questions])
    return pd.DataFrame(records)