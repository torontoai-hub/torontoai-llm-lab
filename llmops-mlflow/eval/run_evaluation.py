from model_config import MODELS_TO_EVALUATE
from evaluator.evaluator import ModelEvaluator

def main():
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(experiment_name="model_comparison_eval")
    
    print(f"Evaluating {len(MODELS_TO_EVALUATE)} model(s):")
    for config in MODELS_TO_EVALUATE:
        print(f"- {config.name}")
    
    # Run comparison
    results = evaluator.compare_models(MODELS_TO_EVALUATE)
    
    # Print summary
    print("\nEvaluation Summary:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        
        # Print basic metrics
        print("\nBasic Metrics:")
        for metric in ["avg_exact_match", "avg_latency", "avg_response_length_ratio"]:
            value = result["metrics"][metric]
            print(f"  {metric}: {value:.3f}")
        
        # Print overall statistics
        print("\nOverall Statistics:")
        print(f"  Total Requests: {result['metrics']['total_requests']}")
        print(f"  Overall Success Rate: {result['metrics']['overall_success_rate']:.3f}")
        print(f"  Error Rate: {result['metrics']['error_rate']:.3f}")
        print(f"  Evaluation Duration: {result['metrics']['evaluation_duration']:.2f}s")
        
        # Print category-wise performance from observability data
        print("\nCategory-wise Performance:")
        if 'observability' in result:
            category_metrics = result['observability']['category_metrics']
            for category, metrics in category_metrics.items():
                print(f"  {category}:")
                print(f"    Success Rate: {metrics['success_rate']:.3f}")
                print(f"    Avg Latency: {metrics['avg_latency']:.3f}s")
                print(f"    Sample Size: {metrics['sample_size']}")
    
    print("\nDetailed results, logs, and visualizations are available:")
    print("1. MLflow UI:")
    print("     mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("2. Detailed Logs:")
    print("     - Response logs: logs/{model_name}_responses_{timestamp}.json")
    print("     - Error logs: logs/{model_name}_errors_{timestamp}.json")
    print("3. Model evaluation logs: model_evaluation.log")

if __name__ == "__main__":
    main()