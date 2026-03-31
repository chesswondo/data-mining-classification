import yaml
import matplotlib.pyplot as plt

from src.data_loader import load_and_preprocess_data
from src.models.decision_tree import DecisionTree
from src.metrics import accuracy_score

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config.yaml")
    
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(
        csv_path=config['data']['path'],
        target_col=config['data']['target_column'],
        drop_cols=config['data'].get('drop_columns', []),
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Hyperparameter range
    max_depths = list(range(1, 16))
    train_scores = []
    test_scores = []
    
    for depth in max_depths:
        tree = DecisionTree(max_depth=depth, min_samples_split=2)
        
        tree.fit(X_train, y_train)
        
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        
        print(f"Depth: {depth:2d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    plt.figure(figsize=(10, 6))
    
    plt.plot(max_depths, train_scores, label='Train Accuracy', marker='o', color='blue')
    plt.plot(max_depths, test_scores, label='Test Accuracy', marker='s', color='orange')
    
    plt.title('Вплив максимальної глибини дерева (max_depth) на точність моделі', fontsize=14)
    plt.xlabel('Максимальна глибина дерева (max_depth)', fontsize=12)
    plt.ylabel('Точність (Accuracy)', fontsize=12)
    plt.xticks(max_depths)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.savefig('tree_hyperparams_analysis.png', dpi=300, bbox_inches='tight')
        
    plt.show()

if __name__ == "__main__":
    main()