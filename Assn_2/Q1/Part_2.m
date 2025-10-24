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

fprintf('Generating datasets...\n');
[D50_train, labels50] = generateData(50, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
[D500_train, labels500] = generateData(500, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
[D5000_train, labels5000] = generateData(5000, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
[D10K_validate, labels10K] = generateData(10000, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
fprintf('Datasets generated.\n\n');

fprintf('========== Part 2(a): Logistic-Linear Models ==========\n');
[w_lin_50, error_lin_50] = trainLogisticLinear(D50_train, labels50, D10K_validate, labels10K);
fprintf('Logistic-Linear with 50 samples: P(error) = %.4f\n', error_lin_50);

[w_lin_500, error_lin_500] = trainLogisticLinear(D500_train, labels500, D10K_validate, labels10K);
fprintf('Logistic-Linear with 500 samples: P(error) = %.4f\n', error_lin_500);

[w_lin_5000, error_lin_5000] = trainLogisticLinear(D5000_train, labels5000, D10K_validate, labels10K);
fprintf('Logistic-Linear with 5000 samples: P(error) = %.4f\n\n', error_lin_5000);

fprintf('========== Part 2(b): Logistic-Quadratic Models ==========\n');
[w_quad_50, error_quad_50] = trainLogisticQuadratic(D50_train, labels50, D10K_validate, labels10K);
fprintf('Logistic-Quadratic with 50 samples: P(error) = %.4f\n', error_quad_50);

[w_quad_500, error_quad_500] = trainLogisticQuadratic(D500_train, labels500, D10K_validate, labels10K);
fprintf('Logistic-Quadratic with 500 samples: P(error) = %.4f\n', error_quad_500);

[w_quad_5000, error_quad_5000] = trainLogisticQuadratic(D5000_train, labels5000, D10K_validate, labels10K);
fprintf('Logistic-Quadratic with 5000 samples: P(error) = %.4f\n\n', error_quad_5000);

datasets = {D50_train, D500_train, D5000_train};
labels_sets = {labels50, labels500, labels5000};
w_lins = {w_lin_50, w_lin_500, w_lin_5000};
w_quads = {w_quad_50, w_quad_500, w_quad_5000};
sample_sizes = [50, 500, 5000];

figure('Position', [100 100 1400 1000]);
for i = 1:3
    subplot(3,2,2*i-1);
    plotDecisionBoundaryLinear(w_lins{i}, datasets{i}, labels_sets{i});
    title(sprintf('Logistic-Linear: %d Training Samples', sample_sizes(i)), 'FontSize', 12);
    
    subplot(3,2,2*i);
    plotDecisionBoundaryQuadratic(w_quads{i}, datasets{i}, labels_sets{i});
    title(sprintf('Logistic-Quadratic: %d Training Samples', sample_sizes(i)), 'FontSize', 12);
end

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

function [w_opt, error_rate] = trainLogisticLinear(X_train, y_train, X_val, y_val)
    z_train = [ones(1, size(X_train,2)); X_train];
    w0 = randn(size(z_train,1), 1) * 0.01;
    options = optimset('Display', 'off', 'MaxIter', 5000, 'TolFun', 1e-8);
    w_opt = fminsearch(@(w) negLogLikelihood(w, z_train, y_train), w0, options);
    z_val = [ones(1, size(X_val,2)); X_val];
    predictions = (1./(1 + exp(-w_opt' * z_val)) > 0.5);
    error_rate = mean(predictions' ~= y_val);
end

function nll = negLogLikelihood(w, z, y)
    scores = w' * z;
    h = 1./(1 + exp(-scores));
    h = max(min(h, 1-1e-15), 1e-15);
    nll = -sum(y' .* log(h) + (1-y') .* log(1-h));
end

function [w_opt, error_rate] = trainLogisticQuadratic(X_train, y_train, X_val, y_val)
    z_train = [ones(1, size(X_train,2)); 
               X_train(1,:); 
               X_train(2,:); 
               X_train(1,:).^2; 
               X_train(1,:).*X_train(2,:); 
               X_train(2,:).^2];
    w0 = randn(size(z_train,1), 1) * 0.01;
    options = optimset('Display', 'off', 'MaxIter', 5000, 'TolFun', 1e-8);
    w_opt = fminsearch(@(w) negLogLikelihood(w, z_train, y_train), w0, options);
    z_val = [ones(1, size(X_val,2)); 
             X_val(1,:); 
             X_val(2,:); 
             X_val(1,:).^2; 
             X_val(1,:).*X_val(2,:); 
             X_val(2,:).^2];
    predictions = (1./(1 + exp(-w_opt' * z_val)) > 0.5);
    error_rate = mean(predictions' ~= y_val);
end

function plotDecisionBoundaryLinear(w, X, labels)
    x1_range = linspace(min(X(1,:))-1, max(X(1,:))+1, 300);
    x2_range = linspace(min(X(2,:))-1, max(X(2,:))+1, 300);
    [X1, X2] = meshgrid(x1_range, x2_range);
    Z = w(1) + w(2)*X1 + w(3)*X2;
    contour(X1, X2, Z, [0 0], 'k-', 'LineWidth', 2);
    hold on;
    idx0 = labels == 0;
    idx1 = labels == 1;
    scatter(X(1,idx0), X(2,idx0), 50, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(X(1,idx1), X(2,idx1), 50, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('x_1', 'FontSize', 11);
    ylabel('x_2', 'FontSize', 11);
    grid on;
    legend('Decision Boundary', 'Class 0', 'Class 1', 'Location', 'best');
    axis equal;
end

function plotDecisionBoundaryQuadratic(w, X, labels)
    x1_range = linspace(min(X(1,:))-1, max(X(1,:))+1, 300);
    x2_range = linspace(min(X(2,:))-1, max(X(2,:))+1, 300);
    [X1, X2] = meshgrid(x1_range, x2_range);
    Z = w(1) + w(2)*X1 + w(3)*X2 + w(4)*X1.^2 + w(5)*X1.*X2 + w(6)*X2.^2;
    contour(X1, X2, Z, [0 0], 'k-', 'LineWidth', 2);
    hold on;
    idx0 = labels == 0;
    idx1 = labels == 1;
    scatter(X(1,idx0), X(2,idx0), 50, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(X(1,idx1), X(2,idx1), 50, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('x_1', 'FontSize', 11);
    ylabel('x_2', 'FontSize', 11);
    grid on;
    legend('Decision Boundary', 'Class 0', 'Class 1', 'Location', 'best');
    axis equal;
end
