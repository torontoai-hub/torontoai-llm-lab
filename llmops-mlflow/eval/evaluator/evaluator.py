import mlflow
import pandas as pd
import time
import json
import logging
from datetime import datetime
from openai import OpenAI
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)

class ModelObservability:
    """Track and log model behavior and performance metrics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.responses: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
    def log_response(self, question: str, answer: str, prediction: str, 
                    category: str, latency: float, success: bool):
        """Log each model response with metadata."""
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "expected_answer": answer,
            "model_response": prediction,
            "category": category,
            "latency_seconds": latency,
            "success": success
        }
        self.responses.append(response_data)
        
    def log_error(self, question: str, error_msg: str, category: str):
        """Log model errors."""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "error": error_msg,
            "category": category
        }
        self.errors.append(error_data)
        logging.error(f"Model error - Category: {category}, Question: {question}, Error: {error_msg}")
        
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics from logged data."""
        total_responses = len(self.responses)
        successful_responses = sum(1 for r in self.responses if r["success"])
        total_latency = sum(r["latency_seconds"] for r in self.responses)
        
        # Category-wise success rates
        category_metrics = {}
        for category in set(r["category"] for r in self.responses):
            category_responses = [r for r in self.responses if r["category"] == category]
            success_rate = sum(1 for r in category_responses if r["success"]) / len(category_responses)
            avg_latency = sum(r["latency_seconds"] for r in category_responses) / len(category_responses)
            category_metrics[category] = {
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "sample_size": len(category_responses)
            }
        
        return {
            "total_requests": total_responses,
            "success_rate": successful_responses / total_responses if total_responses > 0 else 0,
            "avg_latency": total_latency / total_responses if total_responses > 0 else 0,
            "error_rate": len(self.errors) / (total_responses + len(self.errors)),
            "category_metrics": category_metrics,
            "total_errors": len(self.errors),
            "evaluation_duration": (datetime.now() - self.start_time).total_seconds()
        }
    
    def save_logs(self, model_name: str):
        """Save detailed logs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save responses log
        responses_file = f"logs/{model_name}_responses_{timestamp}.json"
        with open(responses_file, 'w') as f:
            json.dump(self.responses, f, indent=2)
            
        # Save errors log
        errors_file = f"logs/{model_name}_errors_{timestamp}.json"
        with open(errors_file, 'w') as f:
            json.dump(self.errors, f, indent=2)
        
        return responses_file, errors_file

def create_test_dataset():
    return pd.DataFrame([
        # Math and Logic
        {"question": "What is 13 + 29?", "answer": "42", "category": "math"},
        {"question": "If x + 5 = 12, what is x?", "answer": "7", "category": "math"},
        {"question": "Sort ascending: 9, 2, 11, 3.", "answer": "2, 3, 9, 11", "category": "logic"},
        {"question": "Complete the sequence: 2, 4, 8, 16, __", "answer": "32", "category": "logic"},
        
        # Knowledge and Facts
        {"question": "Capital of Canada?", "answer": "Ottawa", "category": "knowledge"},
        {"question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare", "category": "knowledge"},
        {"question": "What is the chemical symbol for gold?", "answer": "Au", "category": "knowledge"},
        {"question": "Which planet is known as the Red Planet?", "answer": "Mars", "category": "knowledge"},
        
        # Language Understanding
        {"question": "Translate to French: 'apple'", "answer": "pomme", "category": "language"},
        {"question": "What is the plural of 'child'?", "answer": "children", "category": "language"},
        {"question": "Give an antonym for 'happy'", "answer": "sad", "category": "language"},
        {"question": "Is this sentence in past tense: 'She will dance'?", "answer": "no", "category": "language"},
        
        # Common Sense
        {"question": "Is water wet? Answer yes or no.", "answer": "yes", "category": "common_sense"},
        {"question": "Can birds fly underwater? Answer yes or no.", "answer": "no", "category": "common_sense"},
        {"question": "Do you need an umbrella in the rain? Answer yes or no.", "answer": "yes", "category": "common_sense"},
        {"question": "Is fire cold? Answer yes or no.", "answer": "no", "category": "common_sense"},
        
        # Code Understanding
        {"question": "What does this Python code print? print('Hello' + ' World')", "answer": "Hello World", "category": "coding"},
        {"question": "In Python, what is the value of True and False?", "answer": "False", "category": "coding"},
        {"question": "What is the output of [1,2,3][1] in Python?", "answer": "2", "category": "coding"},
        
        # Instructions Following
        {"question": "Say 'hello' three times with spaces between.", "answer": "hello hello hello", "category": "instructions"},
        {"question": "Write 'cat' backwards.", "answer": "tac", "category": "instructions"},
        {"question": "Count from 1 to 3 using words.", "answer": "one two three", "category": "instructions"},
        
        # Text Analysis
        {"question": "How many words are in: 'The quick brown fox'?", "answer": "4", "category": "analysis"},
        {"question": "Is this statement positive or negative: 'I love sunny days'?", "answer": "positive", "category": "analysis"},
        {"question": "What is the main emotion in: 'She cried with joy'?", "answer": "joy", "category": "analysis"}
    ])

def exact_match(row):
    """Calculate exact match score."""
    return float(str(row["prediction"]).lower().strip() == str(row["answer"]).lower().strip())

def response_length(row):
    """Calculate response length ratio."""
    return len(str(row["prediction"])) / len(str(row["answer"]))

class ModelEvaluator:
    def __init__(self, experiment_name: str = "model_comparison_eval"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
        self.eval_dataset = create_test_dataset()

    def evaluate_model(self, config):
        """Evaluate a single model configuration."""
        logging.info(f"Starting evaluation of model: {config.name}")
        client = OpenAI(base_url=config.url, api_key="ollama")
        observer = ModelObservability()

        with mlflow.start_run(run_name=f"eval_{config.name}") as run:
            # Log configuration
            mlflow.log_params({
                "model_name": config.name,
                "temperature": config.temperature,
                "api_base": config.url,
                "evaluation_timestamp": datetime.now().isoformat(),
            })

            # Get predictions and measure latency
            predictions = []
            latencies = []

            # Process each question
            for idx, row in self.eval_dataset.iterrows():
                question = row["question"]
                expected_answer = row["answer"]
                category = row["category"]
                
                start_time = time.time()
                try:
                    r = client.chat.completions.create(
                        model=config.name,
                        messages=[{"role": "user", "content": question}],
                        temperature=config.temperature,
                    )
                    prediction = r.choices[0].message.content.strip()
                    latency = time.time() - start_time
                    
                    predictions.append(prediction)
                    latencies.append(latency)
                    
                    # Log response with observability
                    success = prediction.lower().strip() == expected_answer.lower().strip()
                    observer.log_response(
                        question=question,
                        answer=expected_answer,
                        prediction=prediction,
                        category=category,
                        latency=latency,
                        success=success
                    )
                    
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"Error processing question: {question} - {error_msg}")
                    observer.log_error(question, error_msg, category)
                    predictions.append("")
                    latencies.append(0)

            # Calculate basic metrics
            df = self.eval_dataset.copy()
            df["prediction"] = predictions
            df["latency"] = latencies
            df["exact_match"] = df.apply(exact_match, axis=1)
            df["response_length_ratio"] = df.apply(response_length, axis=1)

            # Get comprehensive metrics from observer
            observability_metrics = observer.get_summary_metrics()
            
            # Combine all metrics
            metrics = {
                "avg_exact_match": float(df["exact_match"].mean()),
                "avg_latency": float(df["latency"].mean()),
                "avg_response_length_ratio": float(df["response_length_ratio"].mean()),
                "total_requests": observability_metrics["total_requests"],
                "overall_success_rate": observability_metrics["success_rate"],
                "error_rate": observability_metrics["error_rate"],
                "evaluation_duration": observability_metrics["evaluation_duration"]
            }
            
            # Add category-specific metrics
            for category, cat_metrics in observability_metrics["category_metrics"].items():
                metrics[f"{category}_success_rate"] = cat_metrics["success_rate"]
                metrics[f"{category}_avg_latency"] = cat_metrics["avg_latency"]

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Save and log detailed results
            df.to_csv(f"results_{config.name}.csv", index=False)
            mlflow.log_artifact(f"results_{config.name}.csv")
            
            # Save and log observability data
            responses_file, errors_file = observer.save_logs(config.name)
            mlflow.log_artifact(responses_file)
            mlflow.log_artifact(errors_file)
            
            # Log additional MLflow tags
            mlflow.set_tags({
                "total_errors": observability_metrics["total_errors"],
                "evaluation_duration": observability_metrics["evaluation_duration"],
                "test_categories": ", ".join(df["category"].unique()),
                "num_test_cases": len(df)
            })

            return {
                "model_name": config.name,
                "metrics": metrics,
                "results": df,
                "observability": observability_metrics
            }

    def compare_models(self, configs):
        """Run evaluation for all models."""
        results = {}
        for config in configs:
            result = self.evaluate_model(config)
            if result:
                results[config.name] = result
        return results
        
        return results