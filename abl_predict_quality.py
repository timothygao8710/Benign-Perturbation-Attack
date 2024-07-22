import json
import ast
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import *

def load(file_path):
    data = load_data(file_path)

    features, output = [], []
    for row in data:
        output.append(row["is_correct"])
        cur_features = []
        cur_features.append(row["model_prob"][0])
        # cur_features.append(row["entropy"])
        # cur_features.extend(row["model_prob"])
        # diffs = [row["model_prob"][i-1] - row["model_prob"][i] for i in range(1, len(row["model_prob"]))]
        # cur_features.extend(diffs)
        features.append(cur_features)

        print(row)
    return features, output

file_path = './data/medqa_llama'
features, output = load(file_path)

# Convert to numpy arrays
X = np.array(features)
y = np.array(output)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Ridge Classifier': RidgeClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'SVM (RBF kernel)': SVC(kernel='rbf', random_state=42),
    'SVM (Linear kernel)': SVC(kernel='linear', random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Define feature sets for ablation study
feature_sets = {
    'All Features': slice(None),
    # 'Entropy Only': slice(0, 1),
    # 'Model Probabilities Only': slice(1, -len(diffs)),
    # 'Probability Differences Only': slice(-len(diffs), None),
    # 'Entropy + Model Probabilities': slice(0, -len(diffs)),
    # 'Entropy + Probability Differences': np.r_[0, -len(diffs):],
    # 'Model Probabilities + Probability Differences': slice(1, None)
}

# Function to evaluate a classifier
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Cross-Validation Score': np.mean(cv_scores)
    }

# Perform ablation study
results = {}
for clf_name, clf in classifiers.items():
    clf_results = {}
    for feature_set_name, feature_slice in feature_sets.items():
        X_train_subset = X_train_scaled[:, feature_slice]
        X_test_subset = X_test_scaled[:, feature_slice]
        clf_results[feature_set_name] = evaluate_classifier(clf, X_train_subset, X_test_subset, y_train, y_test)
    results[clf_name] = clf_results

# Print results
for clf_name, clf_results in results.items():
    print(f"\n{clf_name}:")
    for feature_set_name, metrics in clf_results.items():
        print(f"  {feature_set_name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

# Identify best performing classifier and feature set
best_accuracy = 0
best_clf = ''
best_feature_set = ''
for clf_name, clf_results in results.items():
    for feature_set_name, metrics in clf_results.items():
        if metrics['Accuracy'] > best_accuracy:
            best_accuracy = metrics['Accuracy']
            best_clf = clf_name
            best_feature_set = feature_set_name

print(f"\nBest performing classifier: {best_clf}")
print(f"Best feature set: {best_feature_set}")
print(f"Best accuracy: {best_accuracy:.4f}")

# Generate a summary table
print("\nSummary Table (Accuracy):")
print("Classifier".ljust(25), end="")
for feature_set in feature_sets.keys():
    print(f"{feature_set[:15]:15}", end="")
print()
for clf_name, clf_results in results.items():
    print(f"{clf_name[:24]:25}", end="")
    for feature_set in feature_sets.keys():
        print(f"{clf_results[feature_set]['Accuracy']:.4f}".ljust(15), end="")
    print()