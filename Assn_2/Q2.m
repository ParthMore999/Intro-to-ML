clc; clear all; close all;

rng(42);

N_train = 100;
N_validate = 1000;

[xTrain, yTrain, xValidate, yValidate] = hw2q2(N_train, N_validate);

fprintf('ML Estimator:\n');

figure('Position', [100 100 1200 500]);
subplot(1,2,1);
scatter3(xTrain(1,:), xTrain(2,:), yTrain, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
zlabel('y', 'FontSize', 12);
title('Training Dataset', 'FontSize', 14);
grid on;

subplot(1,2,2);
scatter3(xValidate(1,:), xValidate(2,:), yValidate, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
zlabel('y', 'FontSize', 12);
title('Validation Dataset', 'FontSize', 14);
grid on;

Z_train = zeros(N_train, 10);
for i = 1:N_train
    Z_train(i, :) = cubicFeatures(xTrain(:,i))';
end

Z_validate = zeros(N_validate, 10);
for i = 1:N_validate
    Z_validate(i, :) = cubicFeatures(xValidate(:,i))';
end

w_ML = (Z_train' * Z_train) \ (Z_train' * yTrain');

y_pred_ML = Z_validate * w_ML;
MSE_ML = mean((yValidate' - y_pred_ML).^2);

fprintf('  Average-squared error: %.4f\n\n', MSE_ML);

fprintf('MAP Estimator:\n');

y_pred_ML_train = Z_train * w_ML;
sigma_squared = var(yTrain' - y_pred_ML_train);

gamma_values = logspace(-6, 4, 150);
MSE_MAP_validate = zeros(length(gamma_values), 1);

for idx = 1:length(gamma_values)
    gamma = gamma_values(idx);
    lambda = sigma_squared / gamma;
    
    w_MAP = (Z_train' * Z_train + lambda * eye(10)) \ (Z_train' * yTrain');
    y_pred_MAP = Z_validate * w_MAP;
    MSE_MAP_validate(idx) = mean((yValidate' - y_pred_MAP).^2);
end

[min_MSE, min_idx] = min(MSE_MAP_validate);
optimal_gamma = gamma_values(min_idx);

fprintf('  Optimal gamma: %.4e\n', optimal_gamma);
fprintf('  Average-squared error: %.4f\n\n', min_MSE);

figure('Position', [100 100 1000 500]);
semilogx(gamma_values, MSE_MAP_validate, 'r-', 'LineWidth', 2);
hold on;
semilogx(optimal_gamma, min_MSE, 'g*', 'MarkerSize', 15, 'LineWidth', 3);
yline(MSE_ML, 'b--', 'LineWidth', 2);
xlabel('γ', 'FontSize', 12);
ylabel('Average Squared Error', 'FontSize', 12);
title('Average Squared Error vs Hyperparameter γ', 'FontSize', 14);
legend('MAP', 'Optimal γ', 'ML', 'Location', 'best');
grid on;

function z = cubicFeatures(x)
    x1 = x(1); 
    x2 = x(2);
    z = [1; x1; x2; x1^2; x1*x2; x2^2; x1^3; x1^2*x2; x1*x2^2; x2^3];
end

function [xTrain, yTrain, xValidate, yValidate] = hw2q2(Ntrain, Nvalidate)
    data = generateData(Ntrain);
    xTrain = data(1:2,:); 
    yTrain = data(3,:);
    
    data = generateData(Nvalidate);
    xValidate = data(1:2,:); 
    yValidate = data(3,:);
end

function x = generateData(N)
    gmmParameters.priors = [.3, .4, .3];
    gmmParameters.meanVectors = [-10 0 10; 0 0 0; 10 0 -10];
    gmmParameters.covMatrices(:,:,1) = [1 0 -3; 0 1 0; -3 0 15];
    gmmParameters.covMatrices(:,:,2) = [8 0 0; 0 .5 0; 0 0 .5];
    gmmParameters.covMatrices(:,:,3) = [1 0 -3; 0 1 0; -3 0 15];
    [x, labels] = generateDataFromGMM(N, gmmParameters);
end

function [x, labels] = generateDataFromGMM(N, gmmParameters)
    priors = gmmParameters.priors;
    meanVectors = gmmParameters.meanVectors;
    covMatrices = gmmParameters.covMatrices;
    n = size(gmmParameters.meanVectors, 1);
    C = length(priors);
    x = zeros(n, N); 
    labels = zeros(1, N);
    
    u = rand(1, N); 
    thresholds = [cumsum(priors), 1];
    
    for l = 1:C
        indl = find(u <= thresholds(l)); 
        Nl = length(indl);
        labels(1, indl) = l * ones(1, Nl);
        u(1, indl) = 1.1 * ones(1, Nl);
        x(:, indl) = mvnrnd(meanVectors(:, l), covMatrices(:, :, l), Nl)';
    end
end
