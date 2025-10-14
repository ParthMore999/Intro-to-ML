clear all; close all; clc;

load('classification_data.mat');

Sigma0_NB = eye(3);
Sigma1_NB = eye(3);
mu0_NB = mu0;
mu1_NB = mu1;

likelihood_0_NB = mvnpdf(data, mu0_NB', Sigma0_NB);
likelihood_1_NB = mvnpdf(data, mu1_NB', Sigma1_NB);
likelihood_ratio_NB = likelihood_1_NB ./ likelihood_0_NB;
likelihood_ratio_NB(isnan(likelihood_ratio_NB)) = 0;
likelihood_ratio_NB(isinf(likelihood_ratio_NB)) = 1e10;

min_ratio = min(likelihood_ratio_NB(likelihood_ratio_NB > 0));
max_ratio = max(likelihood_ratio_NB(isfinite(likelihood_ratio_NB)));
gamma_values_NB = [0, logspace(log10(min_ratio), log10(max_ratio), 1000), inf];
n_thresholds = length(gamma_values_NB);

TPR_NB = zeros(n_thresholds, 1);
FPR_NB = zeros(n_thresholds, 1);
FNR_NB = zeros(n_thresholds, 1);
P_error_NB = zeros(n_thresholds, 1);

for i = 1:n_thresholds
    gamma = gamma_values_NB(i);
    decisions = likelihood_ratio_NB > gamma;
    
    TP = sum(decisions == 1 & labels == 1);
    FP = sum(decisions == 1 & labels == 0);
    FN = sum(decisions == 0 & labels == 1);
    
    TPR_NB(i) = TP / N1;
    FPR_NB(i) = FP / N0;
    FNR_NB(i) = FN / N1;
    P_error_NB(i) = FPR_NB(i) * p0 + FNR_NB(i) * p1;
end

[min_error_NB, min_idx_NB] = min(P_error_NB);
optimal_gamma_NB = gamma_values_NB(min_idx_NB);
optimal_TPR_NB = TPR_NB(min_idx_NB);
optimal_FPR_NB = FPR_NB(min_idx_NB);

load('partA3_results.mat');
load('partA2_results.mat');

figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot(FPR_NB, TPR_NB, 'r-', 'LineWidth', 2);
plot(optimal_FPR, optimal_TPR, 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'b');
plot(optimal_FPR_NB, optimal_TPR_NB, 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');
plot([0 1], [0 1], 'k--', 'LineWidth', 1);

xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Curve Comparison: True Model vs Naive Bayes');
legend('True Model', 'Naive Bayes', ...
       sprintf('True Model Min P(error)=%.4f', min_error), ...
       sprintf('NB Min P(error)=%.4f', min_error_NB), ...
       'Random', 'Location', 'southeast');
grid on;
axis([0 1 0 1]);

save('partB_results.mat', 'TPR_NB', 'FPR_NB', 'P_error_NB', 'min_error_NB', ...
     'optimal_gamma_NB', 'optimal_TPR_NB', 'optimal_FPR_NB');
