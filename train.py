import yaml
import argparse

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
    
    print(f"Data loding from {config['data']['path']}...")
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data(
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
        
        model_instance.fit(X_train, y_train)
        preds = model_instance.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec, rec, f1 = precision_recall_f1_macro(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        results[model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        }
        
        print(f"Results {model_name} with params {model_params}:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("\n")

    # Final comparison
    print("=" * 60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
    print("-" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True)
    for name, metrics in sorted_results:
        print(f"{name:<20} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1']:<10.4f}")

if __name__ == "__main__":
    main()