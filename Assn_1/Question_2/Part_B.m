%% Question 2, Part B: ERM Classification with Asymmetric Loss
clear all; close all; clc;

load('q2_data.mat');

Lambda = [0, 10, 10, 100;
          1, 0, 10, 100;
          1, 1, 0, 100;
          1, 1, 1, 0];

decisions_ERM = zeros(length(labels), 1);

for i = 1:length(labels)
    x = data(i, :)';
    
    likelihood = [mvnpdf(x', mu1', Sigma1);
                  mvnpdf(x', mu2', Sigma2);
                  mvnpdf(x', mu3', Sigma3);
                  mvnpdf(x', mu4', Sigma4)];
    
    posterior = likelihood .* priors';
    
    expected_risk = zeros(4, 1);
    for d = 1:4
        for l = 1:4
            expected_risk(d) = expected_risk(d) + Lambda(d, l) * posterior(l);
        end
    end
    
    [~, decisions_ERM(i)] = min(expected_risk);
end

confusion_matrix_ERM = zeros(4, 4);
for i = 1:4
    for j = 1:4
        confusion_matrix_ERM(i, j) = sum(labels == j & decisions_ERM == i);
    end
end

total_risk = 0;
for i = 1:4
    for j = 1:4
        if i ~= j
            total_risk = total_risk + Lambda(i, j) * confusion_matrix_ERM(i, j);
        end
    end
end
average_risk = total_risk / length(labels);

fprintf('Confusion Matrix (ERM with asymmetric loss):\n');
fprintf('        L=1    L=2    L=3    L=4\n');
for i = 1:4
    fprintf('D=%d   %5d  %5d  %5d  %5d\n', i, ...
            confusion_matrix_ERM(i,1), confusion_matrix_ERM(i,2), ...
            confusion_matrix_ERM(i,3), confusion_matrix_ERM(i,4));
end

fprintf('\nMinimum Expected Risk: %.4f\n', average_risk);

figure;
markers = {'o', 's', '^', 'd'};

for true_class = 1:4
    class_idx = (labels == true_class);
    class_data = data(class_idx, :);
    class_decisions = decisions_ERM(class_idx);
    
    correct_idx = (class_decisions == true_class);
    incorrect_idx = ~correct_idx;
    
    if sum(correct_idx) > 0
        scatter(class_data(correct_idx, 1), class_data(correct_idx, 2), ...
                30, 'g', markers{true_class}, 'filled', 'MarkerEdgeColor', 'k', ...
                'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', 0.6);
        hold on;
    end
    
    if sum(incorrect_idx) > 0
        scatter(class_data(incorrect_idx, 1), class_data(incorrect_idx, 2), ...
                30, 'r', markers{true_class}, 'filled', 'MarkerEdgeColor', 'k', ...
                'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', 0.8);
        hold on;
    end
end

xlabel('X1', 'FontSize', 12);
ylabel('X2', 'FontSize', 12);
title(sprintf('ERM Classification with Asymmetric Loss (Risk: %.2f)', average_risk), 'FontSize', 14);
grid on;
axis equal;
xlim([-6 5]);
ylim([-5 5]);

legend_str = {'Class 1', 'Class 2', 'Class 3', 'Class 4 (Protected)'};
text(-5.5, 4.5, 'Green=Correct, Red=Incorrect', 'FontSize', 10);

save('q2_partB_results.mat', 'decisions_ERM', 'confusion_matrix_ERM', 'average_risk');
