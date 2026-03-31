import yaml
import argparse
import time

from src.data_loader import load_and_preprocess_data
from src.metrics import accuracy_score, confusion_matrix, precision_recall_f1_macro
from src.models.one_rule import OneRule
from src.models.decision_tree import DecisionTree
from src.models.knn import KNN
from src.models.naive_bayes import GaussianNaiveBayes

MODEL_REGISTRY = {
    'knn': KNN,
    'decision_tree': DecisionTree,
    'gaussian_naive_bayes': GaussianNaiveBayes,
    'one_rule': OneRule
}

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Start ML pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print(f"Data loading from {config['data']['path']}...")
    X_train, X_test, y_train, y_test, le, feature_names = load_and_preprocess_data(
        csv_path=config['data']['path'],
        drop_cols=config['data']['drop_columns'],
        target_col=config['data']['target_column'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    print(f"Train set shape: {X_train.shape}, test set shape: {X_test.shape}\n")
    
    results = {}
    
    # Dynamic initialization and training
    for model_name, model_params in config.get('models', {}).items():
        if model_name not in MODEL_REGISTRY:
            print(f"Warning: model {model_name} not found in MODEL_REGISTRY. Skipping...")
            continue
            
        print(f"Training {model_name} with params {model_params}...")
        
        model_class = MODEL_REGISTRY[model_name]
        model_instance = model_class(**(model_params or {}))
        
        # Measure training time
        start_fit = time.perf_counter()
        model_instance.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_fit
        
        # Measure inference time
        start_pred = time.perf_counter()
        preds = model_instance.predict(X_test)
        predict_time = time.perf_counter() - start_pred
        
        acc = accuracy_score(y_test, preds)
        prec, rec, f1 = precision_recall_f1_macro(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        results[model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Fit_Time': fit_time,
            'Predict_Time': predict_time
        }
        
        print(f"Results {model_name} with params {model_params}:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Fit time:  {fit_time:.6f} sec")
        print(f"Pred time: {predict_time:.6f} sec")
        
        # Extract root feature index for Decision Tree to find the most important feature
        if model_name == 'decision_tree' and hasattr(model_instance, 'root') and model_instance.root is not None:
            root_idx = model_instance.root.feature
            root_name = feature_names[root_idx]
            print(f"Root split feature index: {root_idx}")
            print(f"Most important feature (Root): {root_name}")
            
        print("Confusion Matrix:")
        print(cm)
        print("\n")

    # Final comparison
    print("=" * 106)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'Fit Time (s)':<15} | {'Pred Time (s)':<15}")
    print("-" * 106)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True)
    for name, metrics in sorted_results:
        # Re-formatted print to accommodate time metrics cleanly
        print(f"{name:<20} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1']:<10.4f} | {metrics['Fit_Time']:<15.6f} | {metrics['Predict_Time']:<15.6f}")

if __name__ == "__main__":
    main()