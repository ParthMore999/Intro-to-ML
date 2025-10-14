
clear all; close all; clc;

N = 10000;
p0 = 0.65;
p1 = 0.35;

u = rand(1,N) >= p0;
labels = u';
N0 = sum(labels == 0);
N1 = sum(labels == 1);

mu0 = [-1/2; -1/2; -1/2];
Sigma0 = [1, -0.5, 0.3; -0.5, 1, -0.5; 0.3, -0.5, 1];
r0 = mvnrnd(mu0', Sigma0, N0);

mu1 = [1; 1; 1];
Sigma1 = [1, 0.3, -0.2; 0.3, 1, 0.3; -0.2, 0.3, 1];
r1 = mvnrnd(mu1', Sigma1, N1);

data = zeros(N, 3);
data(labels == 0, :) = r0;
data(labels == 1, :) = r1;

save('classification_data.mat', 'data', 'labels', 'N0', 'N1', ...
     'mu0', 'Sigma0', 'mu1', 'Sigma1', 'p0', 'p1');

likelihood_0 = mvnpdf(data, mu0', Sigma0);
likelihood_1 = mvnpdf(data, mu1', Sigma1);
likelihood_ratio = likelihood_1 ./ likelihood_0;
likelihood_ratio(isnan(likelihood_ratio)) = 0;
likelihood_ratio(isinf(likelihood_ratio)) = 1e10;

min_ratio = min(likelihood_ratio(likelihood_ratio > 0));
max_ratio = max(likelihood_ratio(isfinite(likelihood_ratio)));
gamma_values = [0, logspace(log10(min_ratio), log10(max_ratio), 1000), inf];
n_thresholds = length(gamma_values);

TPR = zeros(n_thresholds, 1);
FPR = zeros(n_thresholds, 1);
FNR = zeros(n_thresholds, 1);

for i = 1:n_thresholds
    gamma = gamma_values(i);
    decisions = likelihood_ratio > gamma;
    
    TP = sum(decisions == 1 & labels == 1);
    FP = sum(decisions == 1 & labels == 0);
    FN = sum(decisions == 0 & labels == 1);
    
    TPR(i) = TP / N1;
    FPR(i) = FP / N0;
    FNR(i) = FN / N1;
end

figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Curve - Minimum Expected Risk Classifier');
legend('ROC Curve', 'Random Classifier', 'Location', 'southeast');
grid on;
axis([0 1 0 1]);

save('partA2_results.mat', 'gamma_values', 'TPR', 'FPR', 'FNR', 'likelihood_ratio');
