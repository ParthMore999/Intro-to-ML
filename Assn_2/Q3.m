clc; clear all; close all;

rng(42);

true_pos = [0.5; 0.3];
sigma_x = 0.25;
sigma_y = 0.25;
sigma_noise = 0.3;

fprintf('Vehicle Localization Results:\n');
fprintf('True position: [%.2f, %.2f]\n\n', true_pos(1), true_pos(2));

K_values = [1, 2, 3, 4];

figure('Position', [100 100 1200 900]);

for k_idx = 1:length(K_values)
    K = K_values(k_idx);
    
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:K);
    landmarks = [cos(angles); sin(angles)];
    
    true_distances = zeros(K, 1);
    measurements = zeros(K, 1);
    for i = 1:K
        true_distances(i) = norm(true_pos - landmarks(:,i));
        valid = false;
        while ~valid
            measurements(i) = true_distances(i) + sigma_noise * randn();
            if measurements(i) >= 0
                valid = true;
            end
        end
    end
    
    x_range = linspace(-2, 2, 150);
    y_range = linspace(-2, 2, 150);
    [X, Y] = meshgrid(x_range, y_range);
    J = zeros(size(X));
    
    for row = 1:size(X,1)
        for col = 1:size(X,2)
            x = X(row, col);
            y = Y(row, col);
            
            sum_term = 0;
            for i = 1:K
                dist = sqrt((x - landmarks(1,i))^2 + (y - landmarks(2,i))^2);
                sum_term = sum_term + (measurements(i) - dist)^2 / (sigma_noise^2);
            end
            
            prior_term = x^2/(sigma_x^2) + y^2/(sigma_y^2);
            J(row, col) = sum_term + prior_term;
        end
    end
    
    [min_val, min_idx] = min(J(:));
    map_estimate = [X(min_idx); Y(min_idx)];
    
    subplot(2, 2, k_idx);
    
    v = linspace(min_val, min_val + 30, 20);
    contour(X, Y, J, v, 'LineWidth', 1.5);
    hold on;
    
    plot(true_pos(1), true_pos(2), 'r+', 'MarkerSize', 20, 'LineWidth', 3);
    plot(landmarks(1,:), landmarks(2,:), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'k');
    plot(map_estimate(1), map_estimate(2), 'g*', 'MarkerSize', 15, 'LineWidth', 2);
    
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5);
    
    colorbar;
    
    xlabel('x', 'FontSize', 12);
    ylabel('y', 'FontSize', 12);
    title(sprintf('K = %d Landmarks', K), 'FontSize', 14);
    
    legend('Contours', 'True Position', 'Landmarks', 'MAP Estimate', 'Unit Circle', 'Location', 'best', 'FontSize', 9);
    
    grid on;
    axis equal;
    xlim([-2 2]);
    ylim([-2 2]);
    
    fprintf('K = %d Landmarks:\n', K);
    fprintf('  MAP estimate: [%.3f, %.3f], Error: %.4f\n\n', map_estimate(1), map_estimate(2), norm(map_estimate - true_pos));
end
