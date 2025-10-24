clc; clear all; close all;

priorL0 = 0.6;
priorL1 = 0.4;
m01 = [-0.9; -1.1];
m02 = [0.8; 0.75];
m11 = [-1.1; 0.9];
m12 = [0.9; -0.75];
C = [0.75 0; 0 1.25];
w01 = 0.5; w02 = 0.5;
w11 = 0.5; w12 = 0.5;

rng(42);

[D10K_validate, labels10K] = generateData(10000, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);

[decisions_opt, scores_opt] = optimalClassifier(D10K_validate, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);

conf_matrix = confusionmat(labels10K, double(decisions_opt));
error_rate = 1 - sum(diag(conf_matrix))/sum(conf_matrix(:));

fprintf('Estimated minimum P(error) = %.4f\n', error_rate);

[fpr, tpr, thresholds] = computeROC(scores_opt, labels10K);

opt_decisions_for_roc = (scores_opt > 0);
opt_tp = sum(opt_decisions_for_roc == 1 & labels10K == 1);
opt_fp = sum(opt_decisions_for_roc == 1 & labels10K == 0);
opt_tn = sum(opt_decisions_for_roc == 0 & labels10K == 0);
opt_fn = sum(opt_decisions_for_roc == 0 & labels10K == 1);
opt_fpr = opt_fp / (opt_fp + opt_tn);
opt_tpr = opt_tp / (opt_tp + opt_fn);

figure('Position', [100 100 800 600]);
plot(fpr, tpr, 'b-', 'LineWidth', 2);
hold on;
plot(opt_fpr, opt_tpr, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate', 'FontSize', 12);
title('ROC Curve for Theoretically Optimal Classifier', 'FontSize', 14);
grid on;
legend('ROC Curve', 'Min P(error) Point', 'Random Classifier', 'Location', 'southeast');
xlim([0 1]);
ylim([0 1]);

figure('Position', [100 100 900 700]);
x1_range = linspace(-4, 4, 400);
x2_range = linspace(-4, 4, 400);
[X1, X2] = meshgrid(x1_range, x2_range);
scores_grid = zeros(size(X1));
for i = 1:numel(X1)
    x = [X1(i); X2(i)];
    scores_grid(i) = discriminantScore(x, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
end
contour(X1, X2, scores_grid, [0 0], 'k-', 'LineWidth', 3);
hold on;
idx0 = labels10K == 0;
idx1 = labels10K == 1;
scatter(D10K_validate(1,idx0), D10K_validate(2,idx0), 15, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
scatter(D10K_validate(1,idx1), D10K_validate(2,idx1), 15, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
plot(m01(1), m01(2), 'bs', 'MarkerSize', 15, 'LineWidth', 3);
plot(m02(1), m02(2), 'bs', 'MarkerSize', 15, 'LineWidth', 3);
plot(m11(1), m11(2), 'rs', 'MarkerSize', 15, 'LineWidth', 3);
plot(m12(1), m12(2), 'rs', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
title('Optimal Decision Boundary with Validation Data', 'FontSize', 14);
legend('Decision Boundary', 'Class 0', 'Class 1', 'Class 0 Means', 'Class 1 Means', 'Location', 'best');
grid on;
axis equal;
xlim([-4 4]);
ylim([-4 4]);

function [X, labels] = generateData(N, p0, p1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    labels = zeros(N, 1);
    X = zeros(2, N);
    
    for i = 1:N
        if rand() < p0
            labels(i) = 0;
            if rand() < w01
                X(:,i) = mvnrnd(m01, C)';
            else
                X(:,i) = mvnrnd(m02, C)';
            end
        else
            labels(i) = 1;
            if rand() < w11
                X(:,i) = mvnrnd(m11, C)';
            else
                X(:,i) = mvnrnd(m12, C)';
            end
        end
    end
end

function score = discriminantScore(x, p0, p1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    px_L0 = w01 * mvnpdf(x', m01', C) + w02 * mvnpdf(x', m02', C);
    px_L1 = w11 * mvnpdf(x', m11', C) + w12 * mvnpdf(x', m12', C);
    score = log(px_L1 * p1) - log(px_L0 * p0);
end

function [decisions, scores] = optimalClassifier(X, p0, p1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    N = size(X, 2);
    scores = zeros(N, 1);
    for i = 1:N
        scores(i) = discriminantScore(X(:,i), p0, p1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    end
    decisions = double(scores > 0);
end

function [fpr, tpr, thresholds] = computeROC(scores, true_labels)
    [sorted_scores, idx] = sort(scores, 'descend');
    sorted_labels = true_labels(idx);
    
    n_pos = sum(true_labels == 1);
    n_neg = sum(true_labels == 0);
    
    tp = 0;
    fp = 0;
    
    fpr = zeros(length(scores)+1, 1);
    tpr = zeros(length(scores)+1, 1);
    
    for i = 1:length(scores)
        if sorted_labels(i) == 1
            tp = tp + 1;
        else
            fp = fp + 1;
        end
        fpr(i+1) = fp / n_neg;
        tpr(i+1) = tp / n_pos;
    end
    
    fpr(1) = 0;
    tpr(1) = 0;
    
    thresholds = [inf; sorted_scores];
end
