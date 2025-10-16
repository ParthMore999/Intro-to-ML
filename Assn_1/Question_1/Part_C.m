
clear all; close all; clc;

load('classification_data.mat');

mu0_est = mean(data(labels==0, :))';
mu1_est = mean(data(labels==1, :))';
Sigma0_est = cov(data(labels==0, :));
Sigma1_est = cov(data(labels==1, :));

Sw = (Sigma0_est + Sigma1_est) / 2;
Sb = (mu1_est - mu0_est) * (mu1_est - mu0_est)';

[V, D] = eig(Sb, Sw);
[~, max_idx] = max(diag(D));
w_LDA = V(:, max_idx);
w_LDA = w_LDA / norm(w_LDA);

fprintf('Fisher LDA Results:\n');
fprintf('==================\n');
fprintf('LDA projection vector w_LDA = [%.4f, %.4f, %.4f]^T\n', w_LDA(1), w_LDA(2), w_LDA(3));

projections = data * w_LDA;

tau_values = linspace(min(projections)-1, max(projections)+1, 1000);
n_thresholds = length(tau_values);

TPR_LDA = zeros(n_thresholds, 1);
FPR_LDA = zeros(n_thresholds, 1);
FNR_LDA = zeros(n_thresholds, 1);
P_error_LDA = zeros(n_thresholds, 1);

for i = 1:n_thresholds
    tau = tau_values(i);
    decisions = projections > tau;
    
    TP = sum(decisions == 1 & labels == 1);
    FP = sum(decisions == 1 & labels == 0);
    FN = sum(decisions == 0 & labels == 1);
    
    TPR_LDA(i) = TP / N1;
    FPR_LDA(i) = FP / N0;
    FNR_LDA(i) = FN / N1;
    P_error_LDA(i) = FPR_LDA(i) * p0 + FNR_LDA(i) * p1;
end

[min_error_LDA, min_idx_LDA] = min(P_error_LDA);
optimal_tau = tau_values(min_idx_LDA);
optimal_TPR_LDA = TPR_LDA(min_idx_LDA);
optimal_FPR_LDA = FPR_LDA(min_idx_LDA);

fprintf('Optimal threshold tau = %.4f\n', optimal_tau);
fprintf('Minimum P(error) = %.4f\n', min_error_LDA);
fprintf('TPR at optimum = %.4f\n', optimal_TPR_LDA);
fprintf('FPR at optimum = %.4f\n', optimal_FPR_LDA);

load('partA3_results.mat');
load('partB_results.mat');
load('partA2_results.mat');

figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot(FPR_NB, TPR_NB, 'r-', 'LineWidth', 2);
plot(FPR_LDA, TPR_LDA, 'g-', 'LineWidth', 2);
plot(optimal_FPR, optimal_TPR, 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'b');
plot(optimal_FPR_NB, optimal_TPR_NB, 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');
plot(optimal_FPR_LDA, optimal_TPR_LDA, 'gd', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g');
plot([0 1], [0 1], 'k--', 'LineWidth', 1);

xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Curve Comparison: All Three Classifiers');
legend('True Model', 'Naive Bayes', 'Fisher LDA', ...
       sprintf('True: P(err)=%.4f', min_error), ...
       sprintf('NB: P(err)=%.4f', min_error_NB), ...
       sprintf('LDA: P(err)=%.4f', min_error_LDA), ...
       'Random', 'Location', 'southeast');
grid on;
axis([0 1 0 1]);

fprintf('\nComparison Summary:\n');
fprintf('==================\n');
fprintf('True Model: P(error) = %.4f\n', min_error);
fprintf('Naive Bayes: P(error) = %.4f\n', min_error_NB);
fprintf('Fisher LDA: P(error) = %.4f\n', min_error_LDA);

save('partC_results.mat', 'TPR_LDA', 'FPR_LDA', 'P_error_LDA', 'min_error_LDA', ...
     'optimal_tau', 'optimal_TPR_LDA', 'optimal_FPR_LDA', 'w_LDA');
