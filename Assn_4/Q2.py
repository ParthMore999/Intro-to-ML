import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("QUESTION 2: GMM-BASED IMAGE SEGMENTATION")
print("="*80)

print(f"\nLoading image from Berkeley Segmentation Dataset...")

try:
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg"
    import requests
    from io import BytesIO
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content))
    print(f"Successfully loaded image from URL")
except Exception as e:
    print(f"Failed to load from URL: {e}")
    print("Creating synthetic image for demonstration...")
    img_array = np.random.randint(50, 200, (150, 200, 3), dtype=np.uint8)
    for i in range(3):
        x_center, y_center = np.random.randint(40, 110), np.random.randint(40, 160)
        y, x = np.ogrid[:150, :200]
        mask = (x - x_center)**2 + (y - y_center)**2 <= 30**2
        img_array[mask] = np.random.randint(100, 255, 3)
    img = Image.fromarray(img_array)

img_array = np.array(img)
height, width = img_array.shape[:2]

print(f"Image size: {width}×{height} = {width*height} pixels")

print(f"\nExtracting 5D features [row, col, R, G, B]...")

row_indices, col_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

features = np.column_stack([
    row_indices.ravel(),
    col_indices.ravel(),
    img_array[:, :, 0].ravel(),
    img_array[:, :, 1].ravel(),
    img_array[:, :, 2].ravel()
])

print(f"Feature matrix: {features.shape}")

print(f"\nNormalizing features to [0, 1]...")

features_normalized = features.copy().astype(float)
features_normalized[:, 0] = features[:, 0] / (height - 1) if height > 1 else 0
features_normalized[:, 1] = features[:, 1] / (width - 1) if width > 1 else 0
features_normalized[:, 2] = features[:, 2] / 255.0
features_normalized[:, 3] = features[:, 3] / 255.0
features_normalized[:, 4] = features[:, 4] / 255.0

print(f"Normalized range: [{features_normalized.min():.3f}, {features_normalized.max():.3f}]")

print("\n" + "="*80)
print("GMM MODEL ORDER SELECTION")
print("="*80)

K_range = [2, 3, 4, 5, 6, 7, 8]
k_folds = 5

print(f"\nK-Fold CV: {k_folds} folds | Components tested: {K_range}")

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
cv_scores = []
cv_std = []

for K in K_range:
    fold_scores = []
    
    for train_idx, val_idx in kf.split(features_normalized):
        X_train_fold = features_normalized[train_idx]
        X_val_fold = features_normalized[val_idx]
        
        gmm = GaussianMixture(n_components=K, covariance_type='full',
                             random_state=42, max_iter=100, n_init=3)
        gmm.fit(X_train_fold)
        
        val_log_likelihood = gmm.score(X_val_fold)
        fold_scores.append(val_log_likelihood)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    cv_scores.append(mean_score)
    cv_std.append(std_score)
    
    print(f"K={K} | Mean Val Log-Likelihood: {mean_score:8.4f} ± {std_score:.4f}")

best_K_idx = np.argmax(cv_scores)
best_K = K_range[best_K_idx]
best_score = cv_scores[best_K_idx]

print(f"\nBest K: {best_K} components")
print(f"Best validation log-likelihood: {best_score:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(K_range, cv_scores, yerr=cv_std, marker='o', markersize=8, 
           linewidth=2, capsize=5, capthick=2, color='steelblue', 
           ecolor='gray', label='Validation Log-Likelihood ± Std')
ax.scatter(best_K, best_score, s=400, c='red', marker='*', 
          edgecolors='black', linewidths=2, zorder=10, label=f'Best: K={best_K}')
ax.set_xlabel('Number of GMM Components (K)', fontsize=12)
ax.set_ylabel('Average Validation Log-Likelihood', fontsize=12)
ax.set_title('GMM Model Order Selection via K-Fold Cross-Validation', 
            fontsize=13, fontweight='bold')
ax.set_xticks(K_range)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('gmm_cv_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFitting final GMM with K={best_K} on all pixels...")

final_gmm = GaussianMixture(n_components=best_K, covariance_type='full',
                            random_state=42, max_iter=200, n_init=5)
final_gmm.fit(features_normalized)

print(f"GMM converged: {final_gmm.converged_} (iterations: {final_gmm.n_iter_})")

labels = final_gmm.predict(features_normalized)
labels_image = labels.reshape(height, width)

print(f"\nSegmentation complete:")
unique_labels = np.unique(labels)
for seg_id in unique_labels:
    count = np.sum(labels == seg_id)
    percentage = (count / len(labels)) * 100
    print(f"  Segment {seg_id}: {count:6d} pixels ({percentage:5.1f}%)")

unique_labels = np.unique(labels_image)
n_segments = len(unique_labels)
segment_colors = np.linspace(0, 255, n_segments, dtype=int)
labels_display = np.zeros_like(labels_image, dtype=np.uint8)

for idx, label in enumerate(unique_labels):
    labels_display[labels_image == label] = segment_colors[idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(img_array)
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(labels_display, cmap='nipy_spectral')
axes[1].set_title(f'GMM Segmentation (K={best_K} components)', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('gmm_segmentation_result.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_array)
axes[0].set_title('Original Image', fontsize=13, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(labels_display, cmap='nipy_spectral')
axes[1].set_title(f'Segmentation (K={best_K}, Color)', fontsize=13, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(labels_display, cmap='gray')
axes[2].set_title(f'Segmentation (K={best_K}, Grayscale)', fontsize=13, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('gmm_segmentation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nImage size: {width}×{height} pixels")
print(f"Optimal K: {best_K} components")
print(f"Best log-likelihood: {best_score:.4f}")
print(f"Number of segments: {n_segments}")
print("="*80)
