import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_data
from semantic_uncertainty.calc_entropy import get_entropy_from_probabilities

# Load and preprocess data
def load_and_preprocess(file_path):
    data = load_data(file_path)
    print(data[2497])
    
    features = [
        [
            *row["model_prob"],
            # get_entropy_from_probabilities(row["model_prob"]),
            row["entropy"]
        ]
        for row in data
    ]
    output = [row["is_correct"] for row in data]
    return np.array(features), np.array(output)

def find_unexpected_errors(clf, X_test, y_test, n = 3):
    y_pred = clf.predict_proba(X_test)[:, 0]
    res = []
    diffs = []
    
    for _ in range(n):
        res.append(np.argmax(abs(y_pred - y_test.astype(int))))
        diffs.append(abs(y_pred[res[-1]] - y_test[res[-1]]))
        y_pred[res[-1]] = y_test[res[-1]]
    return (res, diffs)

def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Assuming binary classification for AUROC
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUROC': roc_auc_score(y_test, y_proba),
        "Unexpected Errors": find_unexpected_errors(clf, X_test, y_test)
        
    }

# Run the experiment with or without scaling
def run_experiment(file_path, do_scale=True):
    X, y = load_and_preprocess(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if do_scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=42)
    results = evaluate_classifier(clf, X_train, X_test, y_train, y_test)

    print(f"\nLogistic Regression Results ({'Scaled' if do_scale else 'Unscaled'}):")
    for metric_name, value in results.items():
        if metric_name == 'Unexpected Errors':
            print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: {value:.4f}")

# File path
file_path = '/accounts/projects/binyu/timothygao/Benign-Perturbation-Attack/data/all_entropies'

# Run experiments
run_experiment(file_path, do_scale=True)
run_experiment(file_path, do_scale=False)
