clc; clear all; close all;

rng(42);

angle_true = 2*pi*rand();
radius_true = rand();
true_pos = [radius_true * cos(angle_true); radius_true * sin(angle_true)];

sigma_x = 0.25;
sigma_y = 0.25;
sigma_noise = 0.3;

fprintf('Vehicle Localization Results:\n');
fprintf('True position: [%.2f, %.2f]\n\n', true_pos(1), true_pos(2));

K_values = [1, 2, 3, 4];

x_range = linspace(-2, 2, 200);
y_range = linspace(-2, 2, 200);
[X, Y] = meshgrid(x_range, y_range);

J_all = cell(4, 1);
map_estimates = zeros(2, 4);

for k_idx = 1:length(K_values)
    K = K_values(k_idx);
    
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:K);
    landmarks = [cos(angles); sin(angles)];
    
    measurements = zeros(K, 1);
    for i = 1:K
        true_dist = norm(true_pos - landmarks(:,i));
        valid = false;
        while ~valid
            measurements(i) = true_dist + sigma_noise * randn();
            if measurements(i) >= 0
                valid = true;
            end
        end
    end
    
    J = zeros(size(X));
    for row = 1:size(X,1)
        for col = 1:size(X,2)
            x = X(row, col);
            y = Y(row, col);
            
            likelihood_term = 0;
            for i = 1:K
                dist = sqrt((x - landmarks(1,i))^2 + (y - landmarks(2,i))^2);
                likelihood_term = likelihood_term + (measurements(i) - dist)^2 / (sigma_noise^2);
            end
            
            prior_term = x^2/(sigma_x^2) + y^2/(sigma_y^2);
            J(row, col) = likelihood_term + prior_term;
        end
    end
    
    [min_val, min_idx] = min(J(:));
    map_estimates(:, k_idx) = [X(min_idx); Y(min_idx)];
    
    J_all{k_idx}.J = J;
    J_all{k_idx}.landmarks = landmarks;
    J_all{k_idx}.min_val = min_val;
    
    fprintf('K = %d Landmarks:\n', K);
    fprintf('  MAP estimate: [%.3f, %.3f], Error: %.4f\n\n', ...
        map_estimates(1, k_idx), map_estimates(2, k_idx), norm(map_estimates(:, k_idx) - true_pos));
end

J_min_global = min(cellfun(@(S) S.min_val, J_all));
contour_levels = linspace(J_min_global, J_min_global + 30, 25);

figure('Position', [100 100 1200 900]);

for k_idx = 1:4
    subplot(2, 2, k_idx);
    
    contour(X, Y, J_all{k_idx}.J, contour_levels, 'LineWidth', 1.5);
    hold on;
    colorbar;
    
    plot(true_pos(1), true_pos(2), 'r+', 'MarkerSize', 20, 'LineWidth', 3);
    plot(J_all{k_idx}.landmarks(1,:), J_all{k_idx}.landmarks(2,:), 'ko', ...
        'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'k');
    plot(map_estimates(1, k_idx), map_estimates(2, k_idx), 'g*', ...
        'MarkerSize', 15, 'LineWidth', 2);
    
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5);
    
    xlabel('x', 'FontSize', 12);
    ylabel('y', 'FontSize', 12);
    title(sprintf('K = %d Landmarks', K_values(k_idx)), 'FontSize', 14);
    
    legend('Contours', 'True Position', 'Landmarks', 'MAP Estimate', 'Unit Circle', ...
        'Location', 'best', 'FontSize', 9);
    
    grid on;
    axis equal;
    xlim([-2 2]);
    ylim([-2 2]);
end

sgtitle('MAP Vehicle Localization', 'FontSize', 16);
