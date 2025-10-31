from openai import OpenAI
import pandas as pd
from typing import List, Callable
from .types import ModelConfig

def create_client(config: ModelConfig) -> OpenAI:
    """Create an OpenAI client with the given configuration."""
    return OpenAI(base_url=config.base_url, api_key=config.api_key)

def create_predict_fn(config: ModelConfig) -> Callable[[pd.DataFrame], List[str]]:
    """Create a prediction function for the given model configuration."""
    client = create_client(config)
    
    def predict_fn(batch: pd.DataFrame) -> List[str]:
        outs = []
        questions = batch["question"].tolist()
        
        if config.batch_size and config.batch_size > 1:
            # Batch processing for VLLM
            for i in range(0, len(questions), config.batch_size):
                batch_questions = questions[i:i + config.batch_size]
                messages_list = [[{"role": "user", "content": q}] for q in batch_questions]
                
                try:
                    responses = client.chat.completions.create(
                        model=config.name,
                        messages=messages_list[0],  # VLLM handles batching internally
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    
                    outs.extend([r.choices[0].message.content.strip() for r in responses])
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    outs.extend(["" for _ in batch_questions])
        else:
            # Sequential processing for Ollama
            for q in questions:
                try:
                    r = client.chat.completions.create(
                        model=config.name,
                        messages=[{"role": "user", "content": q}],
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    outs.append(r.choices[0].message.content.strip())
                except Exception as e:
                    print(f"Error with question '{q[:50]}...': {e}")
                    outs.append("")
        
        return outs
    
    return predict_fn