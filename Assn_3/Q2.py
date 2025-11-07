import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import seaborn as sns

np.random.seed(42)

print("="*80)
print("QUESTION 2: GMM MODEL ORDER SELECTION")
print("="*80)

true_means = np.array([
    [0, 0],
    [1.8, 0.8],
    [5, 0],
    [0, 5]
])

true_covariances = np.array([
    [[1.4, 0.4], [0.4, 1.4]],
    [[1.3, -0.3], [-0.3, 1.2]],
    [[1.6, 0.5], [0.5, 1.6]],
    [[1.2, 0.3], [0.3, 1.2]]
])

true_weights = np.array([0.26, 0.25, 0.25, 0.24])

def generate_gmm_data(n_samples):
    samples = []
    
    for _ in range(n_samples):
        component = np.random.choice(4, p=true_weights)
        sample = np.random.multivariate_normal(true_means[component], true_covariances[component])
        samples.append(sample)
    
    return np.array(samples)

def cross_validate_gmm(X, n_components_range, n_folds=10, random_state=None):
    n_samples = X.shape[0]
    actual_folds = min(n_folds, n_samples)
    
    if random_state is None:
        random_state = np.random.randint(10000)
    
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
    cv_scores = {n: [] for n in n_components_range}
    
    for n_comp in n_components_range:
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            
            if len(X_train) < n_comp:
                cv_scores[n_comp].append(-np.inf)
                continue
            
            try:
                gmm = GaussianMixture(
                    n_components=n_comp, 
                    covariance_type='full',
                    max_iter=200,
                    n_init=3,
                    random_state=np.random.randint(10000)
                )
                gmm.fit(X_train)
                val_score = gmm.score(X_val)
                cv_scores[n_comp].append(val_score)
            except:
                cv_scores[n_comp].append(-np.inf)
    
    avg_scores = {n: np.mean(cv_scores[n]) for n in n_components_range}
    best_n = max(avg_scores, key=avg_scores.get)
    
    return best_n, avg_scores

sample_sizes = [10, 100, 1000]
n_components_range = range(1, 11)
n_experiments = 100

print(f"\nTrue model order: 4 components")
print(f"Running {n_experiments} experiments for each sample size...\n")

selection_counts = {n_samples: {n_comp: 0 for n_comp in n_components_range} 
                   for n_samples in sample_sizes}

print("="*80)
print("EXPERIMENTS")
print("="*80)

for exp in range(n_experiments):
    if (exp + 1) % 20 == 0:
        print(f"Progress: {exp + 1}/{n_experiments}")
    
    for n_samples in sample_sizes:
        X = generate_gmm_data(n_samples)
        best_n, _ = cross_validate_gmm(X, n_components_range, n_folds=10, random_state=exp)
        selection_counts[n_samples][best_n] += 1

print(f"Completed: {n_experiments}/{n_experiments}\n")

selection_rates = {}
for n_samples in sample_sizes:
    selection_rates[n_samples] = {n_comp: selection_counts[n_samples][n_comp] / n_experiments 
                                  for n_comp in n_components_range}

print("="*80)
print("RESULTS: SELECTION FREQUENCIES")
print("="*80)
print(f"{'Order':<10}", end="")
for n_samples in sample_sizes:
    print(f"N={n_samples:<12}", end="")
print()
print("-"*80)

for n_comp in n_components_range:
    marker = " *" if n_comp == 4 else "  "
    print(f"{n_comp}{marker:<8}", end="")
    for n_samples in sample_sizes:
        rate = selection_rates[n_samples][n_comp]
        print(f"{rate:<15.2f}", end="")
    print()

print("-"*80)
print("* = True model order\n")

print("Summary:")
for n_samples in sample_sizes:
    correct_count = selection_counts[n_samples][4]
    correct_pct = correct_count / n_experiments * 100
    most_selected = max(selection_rates[n_samples], key=selection_rates[n_samples].get)
    most_pct = selection_rates[n_samples][most_selected] * 100
    print(f"  N={n_samples:4d}: Order 4 selected {correct_count}/100 ({correct_pct:.1f}%), Most selected: {most_selected} ({most_pct:.1f}%)")

print("\n" + "="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n_samples in enumerate(sample_sizes):
    ax = axes[idx]
    
    components = list(n_components_range)
    rates = [selection_rates[n_samples][n] for n in components]
    
    bars = ax.bar(components, rates, color='steelblue', alpha=0.7, edgecolor='black')
    bars[3].set_color('coral')
    
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Selection Rate', fontsize=11)
    ax.set_title(f'N = {n_samples} samples', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(components)

plt.tight_layout()
plt.savefig('q2_selection_rates.png', dpi=300, bbox_inches='tight')
plt.show()

heatmap_data = np.zeros((len(sample_sizes), len(n_components_range)))
for i, n_samples in enumerate(sample_sizes):
    for j, n_comp in enumerate(n_components_range):
        heatmap_data[i, j] = selection_rates[n_samples][n_comp]

plt.figure(figsize=(12, 4))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=list(n_components_range),
            yticklabels=[f'N={n}' for n in sample_sizes],
            cbar_kws={'label': 'Selection Rate'})
plt.xlabel('Number of GMM Components', fontsize=12)
plt.ylabel('Sample Size', fontsize=12)
plt.title('GMM Model Order Selection Frequency (100 experiments)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

X_sample = generate_gmm_data(1000)
plt.figure(figsize=(10, 8))

plt.scatter(X_sample[:, 0], X_sample[:, 1], alpha=0.4, s=20, c='steelblue', label='Generated Data')

for i, (mean, cov) in enumerate(zip(true_means, true_covariances)):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(mean, width, height, angle=angle, 
                     fill=False, edgecolor='red', linewidth=2, linestyle='--')
    plt.gca().add_patch(ellipse)
    plt.plot(mean[0], mean[1], 'r*', markersize=15)

plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.title('True GMM Distribution (4 components with overlapping regions)', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('q2_true_gmm.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*80)
