import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mutual_info_score
from scipy.stats import mannwhitneyu
import seaborn as sns

def load_data(file_path):
    with open(file_path, 'r') as file:
        data_str = json.load(file)
        return ast.literal_eval(data_str)

def analyze_data(data):
    entropies = np.array([item['entropy'] for item in data])
    correctness = np.array([int(item['is_correct']) for item in data])

    print(f"Number of samples: {len(entropies)}")
    print(f"Range of entropy values: {entropies.min()} to {entropies.max()}")
    print(f"Number of unique entropy values: {len(np.unique(entropies))}")
    print(f"Number of correct answers: {np.sum(correctness)}")
    print(f"Number of incorrect answers: {len(correctness) - np.sum(correctness)}")

    if len(entropies) == 0:
        raise ValueError("The entropy array is empty.")

    if len(np.unique(entropies)) < 2:
        raise ValueError("The entropy array contains less than 2 unique values.")

    correlation, p_value = stats.pearsonr(entropies, correctness)

    correct_entropies = entropies[correctness == 1]
    incorrect_entropies = entropies[correctness == 0]
    avg_entropy_correct = np.mean(correct_entropies) if len(correct_entropies) > 0 else np.nan
    avg_entropy_incorrect = np.mean(incorrect_entropies) if len(incorrect_entropies) > 0 else np.nan

    accuracy = np.mean(correctness)

    # Calculate mutual information
    try:
        bins = np.linspace(entropies.min(), entropies.max(), num=min(20, len(np.unique(entropies))))
        digitized_entropies = np.digitize(entropies, bins)
        mi = mutual_info_score(correctness, digitized_entropies)
    except ValueError:
        mi = np.nan
        print("Warning: Could not calculate mutual information.")

    # Calculate effect size (Cohen's d)
    if len(correct_entropies) > 0 and len(incorrect_entropies) > 0:
        pooled_std = np.sqrt((np.std(correct_entropies)**2 + np.std(incorrect_entropies)**2) / 2)
        cohens_d = (avg_entropy_correct - avg_entropy_incorrect) / pooled_std
    else:
        cohens_d = np.nan

    # Perform Mann-Whitney U test
    if len(correct_entropies) > 0 and len(incorrect_entropies) > 0:
        statistic, p_value_mw = mannwhitneyu(correct_entropies, incorrect_entropies)
    else:
        statistic, p_value_mw = np.nan, np.nan

    return {
        'correlation': correlation,
        'p_value': p_value,
        'avg_entropy_correct': avg_entropy_correct,
        'avg_entropy_incorrect': avg_entropy_incorrect,
        'accuracy': accuracy,
        'entropies': entropies,
        'correctness': correctness,
        'mutual_information': mi,
        'cohens_d': cohens_d,
        'mannwhitneyu_statistic': statistic,
        'mannwhitneyu_p_value': p_value_mw
    }

def threshold_analysis(entropies, correctness):
    thresholds = np.linspace(entropies.min(), entropies.max(), 100)
    accuracies = []
    for threshold in thresholds:
        predictions = (entropies < threshold).astype(int)
        accuracy = np.mean(predictions == correctness)
        accuracies.append(accuracy)
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = np.max(accuracies)
    return best_threshold, best_accuracy

def entropy_binning(entropies, correctness, n_bins=10):
    bins = np.linspace(entropies.min(), entropies.max(), n_bins + 1)
    digitized = np.digitize(entropies, bins)
    bin_accuracies = [np.mean(correctness[digitized == i]) for i in range(1, len(bins))]
    return bins[1:], bin_accuracies

def bootstrap_correlation(entropies, correctness, n_iterations=1000, ci=0.95):
    bootstrap_correlations = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, len(entropies), len(entropies))
        bootstrap_correlations.append(stats.pearsonr(entropies[indices], correctness[indices])[0])
    return np.percentile(bootstrap_correlations, [(1-ci)/2 * 100, (1+ci)/2 * 100])

def plot_results(results):
    plt.figure(figsize=(20, 15))

    # Scatter plot
    plt.subplot(231)
    plt.scatter(results['entropies'], results['correctness'], alpha=0.1)
    plt.xlabel('Entropy')
    plt.ylabel('Correctness (0: Incorrect, 1: Correct)')
    plt.title('Scatter Plot: Entropy vs Correctness')
    
    # Add trend line
    z = np.polyfit(results['entropies'], results['correctness'], 1)
    p = np.poly1d(z)
    plt.plot(results['entropies'], p(results['entropies']), "r--", alpha=0.8)
    
    # Add correlation coefficient to the plot
    plt.text(0.05, 0.95, f"Correlation: {results['correlation']:.4f}", transform=plt.gca().transAxes)

    # Histogram
    plt.subplot(232)
    plt.hist([results['entropies'][results['correctness'] == 1], 
              results['entropies'][results['correctness'] == 0]], 
             label=['Correct', 'Incorrect'])
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Entropy for Correct and Incorrect Answers')
    plt.legend()

    # Box plot
    plt.subplot(233)
    sns.boxplot(x=results['correctness'], y=results['entropies'])
    plt.xlabel('Correctness')
    plt.ylabel('Entropy')
    plt.title('Box Plot: Entropy Distribution for Correct and Incorrect Answers')

    # Threshold analysis
    plt.subplot(234)
    best_threshold, _ = threshold_analysis(results['entropies'], results['correctness'])
    plt.hist(results['entropies'], bins=50, alpha=0.5)
    plt.axvline(best_threshold, color='r', linestyle='--')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title(f'Entropy Distribution with Best Threshold ({best_threshold:.2f})')

    # Entropy binning
    plt.subplot(235)
    bins, bin_accuracies = entropy_binning(results['entropies'], results['correctness'])
    plt.plot(bins, bin_accuracies, marker='o')
    plt.xlabel('Entropy')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Entropy Bins')

    # Density plot
    plt.subplot(236)
    sns.kdeplot(data=results['entropies'][results['correctness'] == 0], shade=True, label='Incorrect')
    sns.kdeplot(data=results['entropies'][results['correctness'] == 1], shade=True, label='Correct')
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Density Plot: Entropy Distribution for Correct and Incorrect Answers')
    plt.legend()

    plt.tight_layout()
    plt.savefig('advanced_entropy_analysis_results.png')
    plt.close()

def main():
    file_path = 'acc_v_entropy.json'
    try:
        data = load_data(file_path)
        results = analyze_data(data)

        print(f"\nAnalysis Results:")
        print(f"Correlation between entropy and correctness: {results['correlation']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        print(f"Average entropy for correct answers: {results['avg_entropy_correct']:.4f}")
        print(f"Average entropy for incorrect answers: {results['avg_entropy_incorrect']:.4f}")
        print(f"Overall accuracy: {results['accuracy']:.4f}")
        print(f"Mutual Information: {results['mutual_information']:.4f}")
        print(f"Cohen's d: {results['cohens_d']:.4f}")
        print(f"Mann-Whitney U test statistic: {results['mannwhitneyu_statistic']:.4f}")
        print(f"Mann-Whitney U test p-value: {results['mannwhitneyu_p_value']:.4f}")

        best_threshold, best_accuracy = threshold_analysis(results['entropies'], results['correctness'])
        print(f"Best threshold: {best_threshold:.4f}")
        print(f"Best threshold accuracy: {best_accuracy:.4f}")

        ci_low, ci_high = bootstrap_correlation(results['entropies'], results['correctness'])
        print(f"95% Confidence Interval for correlation: ({ci_low:.4f}, {ci_high:.4f})")

        plot_results(results)
        print("Advanced results plot saved as 'advanced_entropy_analysis_results.png'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data and ensure it contains valid entropy and correctness values.")

if __name__ == "__main__":
    main()