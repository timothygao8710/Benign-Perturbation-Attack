import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import seaborn as sns
from utils import load_data

def analyze_data(data):
    entropies = np.array([item['entropy'] for item in data]).reshape(-1, 1)
    correctness = np.array([int(item['is_correct']) for item in data])

    correlation, p_value = stats.pearsonr(entropies.flatten(), correctness)

    correct_entropies = entropies[correctness == 1]
    incorrect_entropies = entropies[correctness == 0]
    avg_entropy_correct = np.mean(correct_entropies)
    avg_entropy_incorrect = np.mean(incorrect_entropies)

    accuracy = np.mean(correctness)

    return {
        'correlation': correlation,
        'p_value': p_value,
        'avg_entropy_correct': avg_entropy_correct,
        'avg_entropy_incorrect': avg_entropy_incorrect,
        'accuracy': accuracy,
        'entropies': entropies,
        'correctness': correctness
    }

def train_prediction_model(entropies, correctness):
    X_train, X_test, y_train, y_test = train_test_split(entropies, correctness, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report, X_test, y_test, y_pred

def plot_results(results, model, X_test, y_test, y_pred):
    plt.figure(figsize=(20, 15))

    # Density plot
    plt.subplot(221)
    sns.kdeplot(data=results['entropies'][results['correctness'] == 0].flatten(), shade=True, label='Incorrect')
    sns.kdeplot(data=results['entropies'][results['correctness'] == 1].flatten(), shade=True, label='Correct')
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Density Plot: Entropy Distribution for Correct and Incorrect Answers')
    plt.legend()

    # Histogram
    plt.subplot(222)
    plt.hist([results['entropies'][results['correctness'] == 1].flatten(), 
              results['entropies'][results['correctness'] == 0].flatten()], 
             label=['Correct', 'Incorrect'])
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Entropy for Correct and Incorrect Answers')
    plt.legend()

    # Model predictions
    plt.subplot(223)
    plt.scatter(X_test, y_test, color='blue', alpha=0.1, label='Actual')
    plt.scatter(X_test, y_pred, color='red', alpha=0.1, label='Predicted')
    plt.xlabel('Entropy')
    plt.ylabel('Correctness')
    plt.title('Logistic Regression Predictions')
    plt.legend()

    # ROC curve
    plt.subplot(224)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('entropy_analysis_results.png')
    plt.close()

def get_sorted_entropy_lists(data):
    correct_answers = [(i, item['entropy'], item['model_response']) for i, item in enumerate(data) if item['is_correct']]
    incorrect_answers = [(i, item['entropy'], item['model_response']) for i, item in enumerate(data) if not item['is_correct']]
    
    highest_entropy_correct = sorted(correct_answers, key=lambda x: x[1], reverse=True)[:10]
    lowest_entropy_incorrect = sorted(incorrect_answers, key=lambda x: x[1])[:10]
    
    return highest_entropy_correct, lowest_entropy_incorrect

def main():
    file_path = '/accounts/projects/binyu/timothygao/Benign-Perturbation-Attack/data/all_entropies'
    data = load_data(file_path)
    # results = analyze_data(data)

    # print(f"Number of samples: {len(data)}")
    # print(f"Correlation between entropy and correctness: {results['correlation']:.4f}")
    # print(f"P-value: {results['p_value']:.4f}")
    # print(f"Average entropy for correct answers: {results['avg_entropy_correct']:.4f}")
    # print(f"Average entropy for incorrect answers: {results['avg_entropy_incorrect']:.4f}")
    # print(f"Overall accuracy: {results['accuracy']:.4f}")

    # model, model_accuracy, report, X_test, y_test, y_pred = train_prediction_model(results['entropies'], results['correctness'])

    # print("\nPrediction Model Results:")
    # print(f"Model Accuracy: {model_accuracy:.4f}")
    # print("\nClassification Report:")
    # print(report)

    # odds_ratio = np.exp(model.coef_[0])[0]
    # print(f"\nOdds Ratio: {odds_ratio:.4f}")
    # print("Interpretation: For each unit increase in entropy, " 
    #       f"the odds of a correct answer {'increase' if odds_ratio > 1 else 'decrease'} "
    #       f"by a factor of {odds_ratio:.4f}")

    # plot_results(results, model, X_test, y_test, y_pred)
    # print("Results plot saved as 'entropy_analysis_results.png'")
    
    highest_entropy_correct, lowest_entropy_incorrect = get_sorted_entropy_lists(data)
    
    print("\nTop 10 Highest Entropy Correct Answers:")
    for row, entropy, model_response in highest_entropy_correct:
        print(f"Row: {row}, Entropy: {entropy:.4f}, Model Response: {model_response}")
    
    print("\nTop 10 Lowest Entropy Incorrect Answers:")
    for row, entropy, model_response in lowest_entropy_incorrect:
        print(f"Row: {row}, Entropy: {entropy:.4f}, Model Response: {model_response}")

if __name__ == "__main__":
    main()