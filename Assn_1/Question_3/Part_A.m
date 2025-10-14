%% Question 3: Wine Quality - Red Wine
clear all; close all; clc;

red_wine = readtable('winequality-red.csv');

X_wine = table2array(red_wine(:, 1:11));
y_wine = table2array(red_wine(:, 12));

X_wine = normalize(X_wine);

classes_wine = unique(y_wine);
n_classes_wine = length(classes_wine);
n_samples_wine = size(X_wine, 1);

mu_wine = cell(n_classes_wine, 1);
Sigma_wine = cell(n_classes_wine, 1);
priors_wine = zeros(n_classes_wine, 1);

alpha = 0.1;

for i = 1:n_classes_wine
    class_idx = (y_wine == classes_wine(i));
    class_data = X_wine(class_idx, :);
    
    priors_wine(i) = sum(class_idx) / n_samples_wine;
    mu_wine{i} = mean(class_data)';
    
    Sigma_sample = cov(class_data);
    lambda = alpha * trace(Sigma_sample) / size(Sigma_sample, 1);
    Sigma_wine{i} = Sigma_sample + lambda * eye(size(Sigma_sample));
end

decisions_wine = zeros(n_samples_wine, 1);

for i = 1:n_samples_wine
    x = X_wine(i, :)';
    posteriors = zeros(n_classes_wine, 1);
    
    for j = 1:n_classes_wine
        likelihood = mvnpdf(x', mu_wine{j}', Sigma_wine{j});
        posteriors(j) = likelihood * priors_wine(j);
    end
    
    [~, idx] = max(posteriors);
    decisions_wine(i) = classes_wine(idx);
end

confusion_wine = zeros(n_classes_wine, n_classes_wine);
for i = 1:n_classes_wine
    for j = 1:n_classes_wine
        confusion_wine(i,j) = sum(y_wine == classes_wine(j) & decisions_wine == classes_wine(i));
    end
end

error_rate_wine = 1 - sum(diag(confusion_wine)) / n_samples_wine;

fprintf('Wine Quality Error Rate: %.4f\n', error_rate_wine);
fprintf('\nConfusion Matrix:\n');
disp(confusion_wine);

[coeff, score, ~] = pca(X_wine);
figure;
scatter3(score(:,1), score(:,2), score(:,3), 30, y_wine, 'filled');
colorbar;
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('Wine Quality - PCA Visualization');
grid on;
