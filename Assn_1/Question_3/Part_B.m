
clear all; close all; clc;

X_train = load('X_train.txt');
y_train = load('y_train.txt');
X_test = load('X_test.txt');
y_test = load('y_test.txt');

X_har = [X_train; X_test];
y_har = [y_train; y_test];

X_har = normalize(X_har);

classes_har = unique(y_har);
n_classes_har = length(classes_har);
n_samples_har = size(X_har, 1);

mu_har = cell(n_classes_har, 1);
Sigma_har = cell(n_classes_har, 1);
priors_har = zeros(n_classes_har, 1);

alpha = 0.001;

for i = 1:n_classes_har
    class_idx = (y_har == classes_har(i));
    class_data = X_har(class_idx, :);
    
    priors_har(i) = sum(class_idx) / n_samples_har;
    mu_har{i} = mean(class_data)';
    
    Sigma_sample = cov(class_data);
    eigenvals = eig(Sigma_sample);
    eigenvals_pos = eigenvals(eigenvals > 0);
    lambda = alpha * mean(eigenvals_pos);
    Sigma_har{i} = Sigma_sample + lambda * eye(size(Sigma_sample));
end

decisions_har = zeros(n_samples_har, 1);

for i = 1:n_samples_har
    x = X_har(i, :)';
    posteriors = zeros(n_classes_har, 1);
    
    for j = 1:n_classes_har
        likelihood = mvnpdf(x', mu_har{j}', Sigma_har{j});
        posteriors(j) = likelihood * priors_har(j);
    end
    
    [~, idx] = max(posteriors);
    decisions_har(i) = classes_har(idx);
end

confusion_har = zeros(n_classes_har, n_classes_har);
for i = 1:n_classes_har
    for j = 1:n_classes_har
        confusion_har(i,j) = sum(y_har == classes_har(j) & decisions_har == classes_har(i));
    end
end

error_rate_har = 1 - sum(diag(confusion_har)) / n_samples_har;

fprintf('HAR Results:\n');
fprintf('Samples: %d, Features: %d, Classes: %d\n', n_samples_har, size(X_har, 2), n_classes_har);
fprintf('Error Rate: %.4f\n\n', error_rate_har);
fprintf('Confusion Matrix:\n');
disp(confusion_har);

[coeff, score, ~] = pca(X_har);
figure;
scatter3(score(:,1), score(:,2), score(:,3), 10, y_har, 'filled');
colorbar;
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title(sprintf('HAR - PCA Visualization (Error: %.2f%%)', error_rate_har*100));
grid on;
