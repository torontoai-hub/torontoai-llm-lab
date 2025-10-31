import os
import time
import pandas as pd
import mlflow
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from model_config import MODEL_CONFIGS, ModelConfig

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model_comparison_eval")

# Judge model will use the local Ollama instance
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama2")
JUDGE_URL = "http://ollama:11434/v1"
JUDGE_CLIENT = OpenAI(base_url=JUDGE_URL, api_key="ollama")

def create_client(config: ModelConfig) -> OpenAI:
    return OpenAI(base_url=config.base_url, api_key=config.api_key)

eval_categories = {
    "arithmetic": [
        {"question": "What is 13 + 29?", "answer": "42", "category": "arithmetic", "complexity": "simple"},
        {"question": "If you have 5 apples and eat 2, how many do you have left?", "answer": "3", "category": "arithmetic", "complexity": "word_problem"},
        {"question": "Calculate: 15% of 200", "answer": "30", "category": "arithmetic", "complexity": "percentage"},
    ],
    "knowledge": [
        {"question": "Capital of Canada (one word)?", "answer": "Ottawa", "category": "knowledge", "complexity": "simple"},
        {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "category": "knowledge", "complexity": "simple"},
        {"question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare", "category": "knowledge", "complexity": "literature"},
    ],
    "reasoning": [
        {"question": "Sort ascending: 9, 2, 11, 3.", "answer": "2, 3, 9, 11", "category": "reasoning", "complexity": "sorting"},
        {"question": "If all mammals are warm-blooded, and whales are mammals, are whales warm-blooded?", "answer": "yes", "category": "reasoning", "complexity": "logic"},
        {"question": "What comes next in the sequence: 2, 4, 8, 16, ?", "answer": "32", "category": "reasoning", "complexity": "pattern"},
    ],
    "language": [
        {"question": "Translate to French: 'apple'", "answer": "pomme", "category": "language", "complexity": "translation"},
        {"question": "What is the plural of 'child'?", "answer": "children", "category": "language", "complexity": "grammar"},
        {"question": "Give a synonym for 'happy'", "answer": ["joyful", "glad", "delighted", "cheerful"], "category": "language", "complexity": "vocabulary"},
    ],
    "open_ended": [
        {"question": "Explain how a bicycle works in one sentence.", "answer": "A bicycle works by converting the rider's pedaling energy through a chain-drive system to rotate wheels, creating forward motion.", "category": "open_ended", "complexity": "explanation"},
        {"question": "What are three benefits of exercise?", "answer": "Exercise improves cardiovascular health, helps maintain healthy weight, and reduces stress.", "category": "open_ended", "complexity": "listing"},
    ],
    "edge_cases": [
        {"question": "", "answer": "I cannot provide a response without a question.", "category": "edge_cases", "complexity": "empty"},
        {"question": "?????", "answer": "I cannot understand your question. Please provide a clear question.", "category": "edge_cases", "complexity": "unclear"},
    ],
}

# Flatten the dataset
eval_records = []
for category, questions in eval_categories.items():
    eval_records.extend(questions)

eval_dataset = pd.DataFrame(eval_records)

def create_predict_fn(config: ModelConfig):
    client = create_client(config)
    
    def predict_fn(batch: pd.DataFrame) -> list[str]:
        outs = []
        questions = batch["question"].tolist()
        
        if config.batch_size and config.batch_size > 1:
            # Batch processing for VLLM
            for i in range(0, len(questions), config.batch_size):
                batch_questions = questions[i:i + config.batch_size]
                messages_list = [[{"role": "user", "content": q}] for q in batch_questions]
                
                responses = client.chat.completions.create(
                    model=config.name,
                    messages=messages_list[0],  # VLLM handles batching internally
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                
                outs.extend([r.choices[0].message.content.strip() for r in responses])
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
                    print(f"Error with {config.name}: {e}")
                    outs.append("")
        
        return outs
    
    return predict_fn

def _norm(s: str) -> str:
    return "".join(c.lower() for c in str(s) if c.isalnum())

def exact(row): 
    if isinstance(row["answer"], list):
        return float(any(_norm(row["prediction"]) == _norm(ans) for ans in row["answer"]))
    return float(_norm(row["prediction"]) == _norm(row["answer"]))

def contains(row):
    if isinstance(row["answer"], list):
        return float(any(_norm(ans) in _norm(row["prediction"]) for ans in row["answer"]))
    return float(_norm(row["answer"]) in _norm(row["prediction"]))

def token_ratio(row):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pred_tokens = len(tokenizer.encode(row["prediction"]))
    if isinstance(row["answer"], list):
        ref_tokens = min(len(tokenizer.encode(ans)) for ans in row["answer"])
    else:
        ref_tokens = len(tokenizer.encode(row["answer"]))
    ratio = min(pred_tokens / max(1, ref_tokens), 3.0)  # Cap at 3.0
    return ratio

def semantic_similarity(row):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    pred_emb = model.encode(row["prediction"], convert_to_tensor=True)
    if isinstance(row["answer"], list):
        ref_embs = [model.encode(ans, convert_to_tensor=True) for ans in row["answer"]]
        scores = [float(pred_emb @ ref_emb.T) for ref_emb in ref_embs]
        return max(scores)
    ref_emb = model.encode(row["answer"], convert_to_tensor=True)
    return float(pred_emb @ ref_emb.T)

JUDGE_SYSTEM = """You are a strict evaluator for question-answering. Analyze the response based on:
1. Accuracy (0-0.4): How factually correct is the response?
2. Completeness (0-0.3): Does it fully answer all parts of the question?
3. Clarity (0-0.3): Is the response clear and well-structured?

Sum these components for a final score between 0 and 1. Return only the final score as a decimal number."""

def judge_llm(row) -> float:
    prompt = f"QUESTION: {row['question']}\nPREDICTION: {row['prediction']}\nREFERENCE: {row['answer']}\nScore:"
    r = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
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

def measure_latency(row):
    start_time = time.time()
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": row["question"]}],
        temperature=TEMPERATURE,
    )
    return time.time() - start_time

scorers = {
    "exact_match": exact,
    "contains": contains,
    "judge_llm": judge_llm,
    "token_ratio": token_ratio,
    "semantic_similarity": semantic_similarity,
    "response_latency": measure_latency
}

def run_evaluation(config: ModelConfig):
    """Run evaluation for a single model configuration"""
    try:
        from mlflow import genai as mgenai
        with mlflow.start_run(run_name=f"eval_{config.name}_{config.environment}_{int(time.time())}") as run:
            # Log model configuration
            mlflow.log_params({
                "model_name": config.name,
                "environment": config.environment,
                "hardware": config.hardware,
                "temperature": config.temperature,
                "batch_size": config.batch_size,
                "dataset_size": len(eval_dataset),
                "num_categories": len(eval_categories)
            })

            # Evaluate using MLflow genai if available
            results = mgenai.evaluate(data=eval_dataset, predict_fn=qa_predict_fn, scorers=scorers)
            
            # Process and log results
            agg = getattr(results, "scores", None) or getattr(results, "metrics", None)
            if isinstance(agg, dict):
                for k, v in agg.items():
                    try: 
                        mlflow.log_metric(k, float(v))
                    except: 
                        pass

            # Get detailed results table
            table = getattr(results, "table", None)
            if isinstance(table, pd.DataFrame):
                # Log per-category metrics
                for category in eval_dataset["category"].unique():
                    cat_df = table[table["category"] == category]
                    for metric in cat_df.select_dtypes(include=['float64', 'float32', 'int64']).columns:
                        if metric in scorers:
                            mlflow.log_metric(f"{category}_{metric}", float(cat_df[metric].mean()))
                
                # Log per-complexity metrics
                for complexity in eval_dataset["complexity"].unique():
                    complex_df = table[table["complexity"] == complexity]
                    for metric in complex_df.select_dtypes(include=['float64', 'float32', 'int64']).columns:
                        if metric in scorers:
                            mlflow.log_metric(f"{complexity}_{metric}", float(complex_df[metric].mean()))
                
                # Save detailed results
                table.to_csv("genai_eval_results.csv", index=False)
                mlflow.log_artifact("genai_eval_results.csv")
                
                # Generate and log visualizations
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Score distribution by category
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=table, x="category", y="judge_llm")
                plt.xticks(rotation=45)
                plt.title("Score Distribution by Category")
                plt.tight_layout()
                plt.savefig("score_distribution.png")
                mlflow.log_artifact("score_distribution.png")
                plt.close()
                
                # Response latency by complexity
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=table, x="complexity", y="response_latency")
                plt.xticks(rotation=45)
                plt.title("Response Latency by Complexity")
                plt.tight_layout()
                plt.savefig("latency_distribution.png")
                mlflow.log_artifact("latency_distribution.png")
                plt.close()

            print("Run logged. Start UI:\n  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")

    except Exception as e:
        print("mlflow.genai.evaluate failed or not present, falling back:", e)
        with mlflow.start_run(run_name=f"fallback_{MODEL_NAME}_{int(time.time())}"):
            # Perform predictions
            preds = qa_predict_fn(eval_dataset)
            df = eval_dataset.copy()
            df["prediction"] = preds

            # Calculate and log metrics
            for name, fn in scorers.items():
                df[name] = df.apply(fn, axis=1)
                mlflow.log_metric(f"avg_{name}", float(df[name].mean()))
                
                # Log per-category metrics
                for category in df["category"].unique():
                    cat_score = float(df[df["category"] == category][name].mean())
                    mlflow.log_metric(f"{category}_{name}", cat_score)
                
                # Log per-complexity metrics
                for complexity in df["complexity"].unique():
                    complex_score = float(df[df["complexity"] == complexity][name].mean())
                    mlflow.log_metric(f"{complexity}_{name}", complex_score)
            
            # Save detailed results
            df.to_csv("genai_eval_results.csv", index=False)
            mlflow.log_artifact("genai_eval_results.csv")
            
            # Generate and log visualizations (same as above)
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="category", y="judge_llm")
            plt.xticks(rotation=45)
            plt.title("Score Distribution by Category")
            plt.tight_layout()
            plt.savefig("score_distribution.png")
            mlflow.log_artifact("score_distribution.png")
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="complexity", y="response_latency")
            plt.xticks(rotation=45)
            plt.title("Response Latency by Complexity")
            plt.tight_layout()
            plt.savefig("latency_distribution.png")
            mlflow.log_artifact("latency_distribution.png")
            plt.close()

            print("Fallback run logged. Start UI:\n  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")

def compare_models():
    """Run comparative evaluation across all configured models"""
    all_results = {}
    
    # Run evaluations for each model
    for config in MODEL_CONFIGS:
        print(f"\nEvaluating {config.name} on {config.environment} ({config.hardware})...")
        predict_fn = create_predict_fn(config)
        
        try:
            from mlflow import genai as mgenai
            with mlflow.start_run(run_name=f"eval_{config.name}_{config.environment}_{int(time.time())}") as run:
                # Log model configuration
                mlflow.log_params({
                    "model_name": config.name,
                    "environment": config.environment,
                    "hardware": config.hardware,
                    "temperature": config.temperature,
                    "batch_size": config.batch_size,
                    "dataset_size": len(eval_dataset),
                    "num_categories": len(eval_categories)
                })
                
                # Run evaluation
                results = mgenai.evaluate(data=eval_dataset, predict_fn=predict_fn, scorers=scorers)
                
                # Store results for comparison
                all_results[config.name] = results
                
                # Log metrics and artifacts
                if hasattr(results, 'table'):
                    df = results.table
                    
                    # Log per-category metrics
                    for category in df["category"].unique():
                        cat_df = df[df["category"] == category]
                        for metric in [col for col in cat_df.columns if col in scorers]:
                            mlflow.log_metric(f"{category}_{metric}", float(cat_df[metric].mean()))
                    
                    # Save detailed results
                    results_file = f"results_{config.name}.csv"
                    df.to_csv(results_file, index=False)
                    mlflow.log_artifact(results_file)
                    
                    # Generate visualizations
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    # Score distribution
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(data=df, x="category", y="judge_llm")
                    plt.xticks(rotation=45)
                    plt.title(f"Score Distribution by Category - {config.name}")
                    plt.tight_layout()
                    plot_file = f"scores_{config.name}.png"
                    plt.savefig(plot_file)
                    mlflow.log_artifact(plot_file)
                    plt.close()
                    
                    # Latency analysis
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(data=df, x="complexity", y="response_latency")
                    plt.xticks(rotation=45)
                    plt.title(f"Response Latency by Complexity - {config.name}")
                    plt.tight_layout()
                    plot_file = f"latency_{config.name}.png"
                    plt.savefig(plot_file)
                    mlflow.log_artifact(plot_file)
                    plt.close()
        
        except Exception as e:
            print(f"Error evaluating {config.name}: {e}")
            continue
    
    # Generate comparative analysis
    if len(all_results) > 1:
        with mlflow.start_run(run_name=f"comparative_analysis_{int(time.time())}"):
            compare_df = pd.DataFrame()
            
            for config in MODEL_CONFIGS:
                if config.name in all_results:
                    results = all_results[config.name]
                    if hasattr(results, 'table'):
                        df = results.table
                        model_metrics = {
                            'model': config.name,
                            'environment': config.environment,
                            'hardware': config.hardware
                        }
                        
                        # Calculate aggregate metrics
                        for metric in [col for col in df.columns if col in scorers]:
                            model_metrics[f'avg_{metric}'] = df[metric].mean()
                            model_metrics[f'std_{metric}'] = df[metric].std()
                        
                        compare_df = pd.concat([compare_df, pd.DataFrame([model_metrics])])
            
            # Save comparison results
            compare_df.to_csv("model_comparison.csv", index=False)
            mlflow.log_artifact("model_comparison.csv")
            
            # Generate comparison visualizations
            plt.figure(figsize=(12, 6))
            sns.barplot(data=compare_df, x='model', y='avg_judge_llm')
            plt.title("Model Performance Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("model_comparison.png")
            mlflow.log_artifact("model_comparison.png")
            plt.close()
            
            # Latency comparison
            plt.figure(figsize=(12, 6))
            sns.barplot(data=compare_df, x='model', y='avg_response_latency')
            plt.title("Model Latency Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("latency_comparison.png")
            mlflow.log_artifact("latency_comparison.png")
            plt.close()

if __name__ == "__main__":
    compare_models()
