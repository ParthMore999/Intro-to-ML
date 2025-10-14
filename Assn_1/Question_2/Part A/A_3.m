%% Question 2, Part A.3: Visualization
clear all; close all; clc;

load('q2_data.mat');
load('q2_classification_results.mat');

figure('Position', [100, 100, 800, 600]);
markers = {'o', 's', '^', 'd'};

for true_class = 1:4
    class_idx = (labels == true_class);
    class_data = data(class_idx, :);
    class_decisions = decisions(class_idx);
    
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
title(sprintf('MAP Classification Results (Error Rate: %.2f%%)', error_rate*100), 'FontSize', 14);
grid on;
axis equal;
xlim([-6 5]);
ylim([-5 5]);

legend_entries = {};
legend_handles = [];
for i = 1:4
    h = scatter(NaN, NaN, 60, 'g', markers{i}, 'filled', 'MarkerEdgeColor', 'k');
    legend_handles = [legend_handles, h];
    legend_entries{end+1} = sprintf('Class %d', i);
end
h_correct = scatter(NaN, NaN, 60, 'g', 'o', 'filled', 'MarkerEdgeColor', 'k');
h_incorrect = scatter(NaN, NaN, 60, 'r', 'o', 'filled', 'MarkerEdgeColor', 'k');
legend_handles = [legend_handles, h_correct, h_incorrect];
legend_entries{end+1} = 'Correct';
legend_entries{end+1} = 'Incorrect';

legend(legend_handles, legend_entries, 'Location', 'northeastoutside', 'FontSize', 10);

saveas(gcf, 'q2_partA_visualization_fixed.png');
