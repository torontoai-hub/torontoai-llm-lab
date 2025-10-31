import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
import mlflow

def plot_score_distribution(df: pd.DataFrame, model_name: str, metric: str = "judge_llm") -> str:
    """Plot score distribution by category."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="category", y=metric)
    plt.xticks(rotation=45)
    plt.title(f"Score Distribution by Category - {model_name}")
    plt.tight_layout()
    filename = f"scores_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_latency_distribution(df: pd.DataFrame, model_name: str) -> str:
    """Plot latency distribution by complexity."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="complexity", y="response_latency")
    plt.xticks(rotation=45)
    plt.title(f"Response Latency by Complexity - {model_name}")
    plt.tight_layout()
    filename = f"latency_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_model_comparison(compare_df: pd.DataFrame, metric: str = "avg_judge_llm") -> str:
    """Plot model comparison for a specific metric."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=compare_df, x='model', y=metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"comparison_{metric}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def generate_model_visualizations(df: pd.DataFrame, model_name: str, run: Optional[mlflow.ActiveRun] = None):
    """Generate and optionally log all visualizations for a model."""
    files = [
        plot_score_distribution(df, model_name),
        plot_latency_distribution(df, model_name)
    ]
    
    if run:
        for file in files:
            mlflow.log_artifact(file)
    
    return files

def generate_comparison_visualizations(compare_df: pd.DataFrame, run: Optional[mlflow.ActiveRun] = None):
    """Generate and optionally log comparison visualizations."""
    files = [
        plot_model_comparison(compare_df, "avg_judge_llm"),
        plot_model_comparison(compare_df, "avg_response_latency")
    ]
    
    if run:
        for file in files:
            mlflow.log_artifact(file)
    
    return files