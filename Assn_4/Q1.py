import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("QUESTION 1: SVM AND MLP CLASSIFICATION")
print("="*80)

def generate_data(n_samples, r_neg=2, r_pos=4, sigma=1):
    n_per_class = n_samples // 2
    
    theta_neg = np.random.uniform(-np.pi, np.pi, n_per_class)
    noise_neg = np.random.normal(0, sigma, (n_per_class, 2))
    X_neg = r_neg * np.column_stack([np.cos(theta_neg), np.sin(theta_neg)]) + noise_neg
    
    theta_pos = np.random.uniform(-np.pi, np.pi, n_per_class)
    noise_pos = np.random.normal(0, sigma, (n_per_class, 2))
    X_pos = r_pos * np.column_stack([np.cos(theta_pos), np.sin(theta_pos)]) + noise_pos
    
    X = np.vstack([X_neg, X_pos])
    y = np.hstack([-np.ones(n_per_class), np.ones(n_per_class)])
    
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]

print("\nGenerating data...")
X_train, y_train = generate_data(n_samples=1000, r_neg=2, r_pos=4, sigma=1)
X_test, y_test = generate_data(n_samples=10000, r_neg=2, r_pos=4, sigma=1)
print(f"Training: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

plt.figure(figsize=(9, 9))
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
           c='blue', alpha=0.6, s=30, label='Class -1 (r=2)', edgecolors='k', linewidths=0.3)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
           c='red', alpha=0.6, s=30, label='Class +1 (r=4)', edgecolors='k', linewidths=0.3)
plt.xlabel('x₁', fontsize=13)
plt.ylabel('x₂', fontsize=13)
plt.title('Training Data: Two Overlapping Concentric Disks', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('training_data.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("PART 1: SVM WITH GAUSSIAN (RBF) KERNEL")
print("="*80)

gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
C_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
k_folds = 10

print(f"\nK-Fold CV: {k_folds} folds | Gamma: {len(gamma_values)} values | C: {len(C_values)} values")

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
svm_cv_results = np.zeros((len(gamma_values), len(C_values)))

for i, gamma in enumerate(gamma_values):
    for j, C in enumerate(C_values):
        fold_accuracies = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            svm = SVC(kernel='rbf', gamma=gamma, C=C, random_state=42)
            svm.fit(X_train_fold, y_train_fold)
            y_val_pred = svm.predict(X_val_fold)
            fold_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
        
        svm_cv_results[i, j] = np.mean(fold_accuracies)

best_idx = np.unravel_index(np.argmax(svm_cv_results), svm_cv_results.shape)
best_gamma_svm = gamma_values[best_idx[0]]
best_C_svm = C_values[best_idx[1]]
best_cv_acc_svm = svm_cv_results[best_idx]

print(f"\nSVM Cross-Validation Results:")
print(f"  Best: γ={best_gamma_svm}, C={best_C_svm} | CV Acc: {best_cv_acc_svm:.4f} | CV Error: {1-best_cv_acc_svm:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im = axes[0].imshow(svm_cv_results, interpolation='nearest', cmap='viridis', aspect='auto')
axes[0].set_xticks(range(len(C_values)))
axes[0].set_yticks(range(len(gamma_values)))
axes[0].set_xticklabels([f'{c}' for c in C_values])
axes[0].set_yticklabels([f'{g}' for g in gamma_values])
axes[0].set_xlabel('C (Box Constraint)', fontsize=12)
axes[0].set_ylabel('Gamma (Kernel Width)', fontsize=12)
axes[0].set_title('SVM Cross-Validation Accuracy Heatmap', fontsize=13, fontweight='bold')

for i in range(len(gamma_values)):
    for j in range(len(C_values)):
        text_color = 'white' if svm_cv_results[i, j] < 0.85 else 'black'
        axes[0].text(j, i, f'{svm_cv_results[i, j]:.3f}',
                    ha="center", va="center", color=text_color, fontsize=8)

axes[0].scatter(best_idx[1], best_idx[0], marker='*', s=400, 
               c='red', edgecolors='white', linewidths=2, zorder=10)
plt.colorbar(im, ax=axes[0], label='Accuracy')

for i, gamma in enumerate(gamma_values):
    axes[1].plot(C_values, svm_cv_results[i, :], marker='o', label=f'γ={gamma}', linewidth=1.5)
axes[1].set_xlabel('C (Box Constraint)', fontsize=12)
axes[1].set_ylabel('Cross-Validation Accuracy', fontsize=12)
axes[1].set_title('SVM CV Accuracy vs C for Different Gamma', fontsize=13, fontweight='bold')
axes[1].set_xscale('log')
axes[1].legend(fontsize=9, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_cv_results.png', dpi=300, bbox_inches='tight')
plt.close()

final_svm = SVC(kernel='rbf', gamma=best_gamma_svm, C=best_C_svm, random_state=42)
final_svm.fit(X_train, y_train)

y_pred_svm_test = final_svm.predict(X_test)
test_acc_svm = accuracy_score(y_test, y_pred_svm_test)
test_error_svm = 1 - test_acc_svm

print(f"\nSVM Test Performance:")
print(f"  Accuracy: {test_acc_svm:.4f} | Error: {test_error_svm:.4f} | Support Vectors: {final_svm.n_support_.sum()}")

fig, ax = plt.subplots(figsize=(11, 10))

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

Z = final_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu', levels=[-1.5, 0, 1.5])
ax.contour(xx, yy, Z, colors='black', linewidths=2.5, levels=[0])

sample_idx = np.random.choice(len(X_test), 2000, replace=False)
X_test_sample = X_test[sample_idx]
y_test_sample = y_test[sample_idx]

ax.scatter(X_test_sample[y_test_sample == -1, 0], X_test_sample[y_test_sample == -1, 1],
          c='blue', alpha=0.6, s=25, label='Class -1', edgecolors='k', linewidths=0.3)
ax.scatter(X_test_sample[y_test_sample == 1, 0], X_test_sample[y_test_sample == 1, 1],
          c='red', alpha=0.6, s=25, label='Class +1', edgecolors='k', linewidths=0.3)

ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title(f'SVM Decision Boundary on Test Set\nAccuracy: {test_acc_svm:.4f}, Error: {test_error_svm:.4f}', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("PART 2: MLP WITH SINGLE HIDDEN LAYER (QUADRATIC ACTIVATION)")
print("="*80)

class MLP_Quadratic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_Quadratic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = x ** 2
        x = self.fc2(x)
        return x

def train_mlp(model, train_loader, epochs=100, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_mlp(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.numpy() == y).mean()
    return accuracy

hidden_units_list = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

print(f"\nK-Fold CV: {k_folds} folds | Hidden units: {hidden_units_list}")

y_train_binary = (y_train + 1) // 2
y_test_binary = (y_test + 1) // 2

mlp_cv_results = []
mlp_cv_stds = []

for hidden_dim in hidden_units_list:
    fold_accuracies = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train_binary[train_idx], y_train_binary[val_idx]
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_fold), 
                                     torch.LongTensor(y_train_fold))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = MLP_Quadratic(input_dim=2, hidden_dim=hidden_dim, output_dim=2)
        model = train_mlp(model, train_loader, epochs=100, lr=0.01)
        
        val_accuracy = evaluate_mlp(model, X_val_fold, y_val_fold)
        fold_accuracies.append(val_accuracy)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mlp_cv_results.append(mean_acc)
    mlp_cv_stds.append(std_acc)

mlp_cv_results = np.array(mlp_cv_results)
mlp_cv_stds = np.array(mlp_cv_stds)

best_hidden_idx = np.argmax(mlp_cv_results)
best_hidden_units = hidden_units_list[best_hidden_idx]
best_cv_acc_mlp = mlp_cv_results[best_hidden_idx]

print(f"\nMLP Cross-Validation Results:")
print(f"  Best: {best_hidden_units} hidden units | CV Acc: {best_cv_acc_mlp:.4f} | CV Error: {1-best_cv_acc_mlp:.4f}")

fig, ax = plt.subplots(figsize=(12, 7))

ax.errorbar(hidden_units_list, mlp_cv_results, yerr=mlp_cv_stds,
           marker='o', markersize=8, linewidth=2, capsize=5, capthick=2,
           color='steelblue', ecolor='gray', label='CV Accuracy ± Std')

ax.scatter(best_hidden_units, best_cv_acc_mlp,
          s=400, c='red', marker='*', edgecolors='black', linewidths=2,
          label=f'Best: {best_hidden_units} perceptrons', zorder=10)

ax.set_xlabel('Number of Hidden Layer Perceptrons', fontsize=12)
ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
ax.set_title('MLP Cross-Validation: Hidden Layer Size Selection\n(Quadratic Activation Function)', 
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('mlp_cv_results.png', dpi=300, bbox_inches='tight')
plt.close()

train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                             torch.LongTensor(y_train_binary))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

final_mlp = MLP_Quadratic(input_dim=2, hidden_dim=best_hidden_units, output_dim=2)
final_mlp = train_mlp(final_mlp, train_loader, epochs=100, lr=0.01)

test_acc_mlp = evaluate_mlp(final_mlp, X_test, y_test_binary)
test_error_mlp = 1 - test_acc_mlp

print(f"\nMLP Test Performance:")
print(f"  Accuracy: {test_acc_mlp:.4f} | Error: {test_error_mlp:.4f}")

final_mlp.eval()
with torch.no_grad():
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    Z_mlp = final_mlp(grid_tensor)
    _, Z_mlp = torch.max(Z_mlp, 1)
    Z_mlp = Z_mlp.numpy().reshape(xx.shape)

fig, ax = plt.subplots(figsize=(11, 10))

ax.contourf(xx, yy, Z_mlp, alpha=0.25, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
ax.contour(xx, yy, Z_mlp, colors='black', linewidths=2.5, levels=[0.5])

ax.scatter(X_test_sample[y_test_sample == -1, 0], X_test_sample[y_test_sample == -1, 1],
          c='blue', alpha=0.6, s=25, label='Class -1', edgecolors='k', linewidths=0.3)
ax.scatter(X_test_sample[y_test_sample == 1, 0], X_test_sample[y_test_sample == 1, 1],
          c='red', alpha=0.6, s=25, label='Class +1', edgecolors='k', linewidths=0.3)

ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title(f'MLP Decision Boundary on Test Set\nAccuracy: {test_acc_mlp:.4f}, Error: {test_error_mlp:.4f}', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('mlp_decision_boundary.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print("\nSVM (Gaussian/RBF Kernel):")
print(f"  Best: γ={best_gamma_svm}, C={best_C_svm}")
print(f"  CV Acc: {best_cv_acc_svm:.4f} | Test Acc: {test_acc_svm:.4f} | Test Error: {test_error_svm:.4f}")

print("\nMLP (Quadratic Activation):")
print(f"  Best: {best_hidden_units} hidden units")
print(f"  CV Acc: {best_cv_acc_mlp:.4f} | Test Acc: {test_acc_mlp:.4f} | Test Error: {test_error_mlp:.4f}")

if test_error_svm < test_error_mlp:
    print(f"\n→ SVM outperforms MLP by {(test_error_mlp - test_error_svm)*100:.2f}% in error rate")
else:
    print(f"\n→ MLP outperforms SVM by {(test_error_svm - test_error_mlp)*100:.2f}% in error rate")
print("="*80)
