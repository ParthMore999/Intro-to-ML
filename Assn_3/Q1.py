import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("QUESTION 1: MLP CLASSIFICATION")
print("="*80)

class_priors = np.array([0.25, 0.25, 0.25, 0.25])

means = [
    np.array([0, 0, 0]),
    np.array([2.5, 2, 0]),
    np.array([2, 0, 2.5]),
    np.array([0, 2.5, 2])
]

covariances = [
    np.array([[1.5, 0.3, 0.3], [0.3, 1.5, 0.3], [0.3, 0.3, 1.5]]),
    np.array([[1.3, -0.2, 0.2], [-0.2, 1.3, -0.2], [0.2, -0.2, 1.3]]),
    np.array([[1.6, 0.3, -0.2], [0.3, 1.6, 0.3], [-0.2, 0.3, 1.6]]),
    np.array([[1.4, -0.2, 0.3], [-0.2, 1.4, -0.2], [0.3, -0.2, 1.4]])
]

def generate_data(n_samples):
    samples = []
    labels = []
    
    for _ in range(n_samples):
        class_idx = np.random.choice(4, p=class_priors)
        sample = np.random.multivariate_normal(means[class_idx], covariances[class_idx])
        samples.append(sample)
        labels.append(class_idx)
    
    return np.array(samples), np.array(labels)

def map_classifier(X):
    n_samples = X.shape[0]
    posteriors = np.zeros((n_samples, 4))
    
    for c in range(4):
        likelihood = multivariate_normal.pdf(X, mean=means[c], cov=covariances[c])
        posteriors[:, c] = likelihood * class_priors[c]
    
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    return np.argmax(posteriors, axis=1)

class MLP(nn.Module):
    def __init__(self, n_perceptrons):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, n_perceptrons)
        self.output = nn.Linear(n_perceptrons, 4)
        self.activation = nn.ELU()
        
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return torch.softmax(x, dim=1)

def train_mlp(X_train, y_train, n_perceptrons, n_epochs=400, n_inits=5):
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    best_model = None
    best_loss = float('inf')
    
    for init in range(n_inits):
        model = MLP(n_perceptrons)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model
    
    return best_model

def evaluate_mlp(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()
    
    error = np.mean(predictions != y_test)
    return error

def cross_validate(X_train, y_train, perceptron_range, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_errors = {p: [] for p in perceptron_range}
    
    for p in perceptron_range:
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = train_mlp(X_tr, y_tr, p, n_epochs=250, n_inits=3)
            error = evaluate_mlp(model, X_val, y_val)
            cv_errors[p].append(error)
    
    avg_errors = {p: np.mean(cv_errors[p]) for p in perceptron_range}
    best_p = min(avg_errors, key=avg_errors.get)
    
    return best_p, avg_errors

train_sizes = [100, 500, 1000, 5000, 10000]
X_test, y_test = generate_data(100000)

map_predictions = map_classifier(X_test)
map_error = np.mean(map_predictions != y_test)
print(f"\nTheoretical Optimal MAP Classifier Error: {map_error:.4f} ({map_error*100:.2f}%)\n")

perceptron_range = [2, 4, 8, 12, 16, 24, 32]
results = {}

print("="*80)
print("TRAINING MLPS")
print("="*80)

for idx, n_train in enumerate(train_sizes, 1):
    print(f"\n[{idx}/{len(train_sizes)}] N = {n_train}")
    X_train, y_train = generate_data(n_train)
    
    print(f"  Cross-validation...", end=' ')
    best_p, cv_errors = cross_validate(X_train, y_train, perceptron_range)
    print(f"Best P = {best_p}")
    
    print(f"  Training final model...", end=' ')
    final_model = train_mlp(X_train, y_train, best_p, n_epochs=400, n_inits=8)
    test_error = evaluate_mlp(final_model, X_test, y_test)
    print(f"Test Error = {test_error:.4f}")
    
    results[n_train] = {
        'best_p': best_p,
        'test_error': test_error,
        'cv_errors': cv_errors
    }

plt.figure(figsize=(10, 6))
train_sizes_list = list(results.keys())
test_errors = [results[n]['test_error'] for n in train_sizes_list]

plt.semilogx(train_sizes_list, test_errors, 'bo-', linewidth=2, markersize=10, label='MLP Classifier')
plt.axhline(y=map_error, color='r', linestyle='--', linewidth=2, label='Theoretical MAP Classifier')
plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Test Error Probability', fontsize=12)
plt.title('MLP Classification Performance vs Training Set Size', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('q1_performance.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for n_train in train_sizes:
    cv_errors = results[n_train]['cv_errors']
    perceptrons = list(cv_errors.keys())
    errors = list(cv_errors.values())
    plt.plot(perceptrons, errors, 'o-', linewidth=2, markersize=6, label=f'N = {n_train}')

plt.axhline(y=map_error, color='black', linestyle='--', linewidth=2, label='Min. Pr(error)')
plt.xlabel('Number of Perceptrons (P)', fontsize=12)
plt.ylabel('Cross-Validation Pr(error)', fontsize=12)
plt.title('Number of Perceptrons vs Cross-Validation Error', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('q1_cv_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nOptimal MAP Classifier: {map_error:.4f} ({map_error*100:.2f}%)\n")
print(f"{'Training Samples':<20} {'Best P':<12} {'Test Error':<15} {'Error %':<12}")
print("-"*80)
for n_train in train_sizes:
    best_p = results[n_train]['best_p']
    test_error = results[n_train]['test_error']
    print(f"{n_train:<20} {best_p:<12} {test_error:<15.4f} {test_error*100:<12.2f}")
print("="*80)
