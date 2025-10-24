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

fprintf('Generating Datasets...\n');
[D50_train, labels50] = generateData(50, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
fprintf('D50_train: %d samples generated\n', size(D50_train, 2));

[D500_train, labels500] = generateData(500, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
fprintf('D500_train: %d samples generated\n', size(D500_train, 2));

[D5000_train, labels5000] = generateData(5000, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
fprintf('D5000_train: %d samples generated\n', size(D5000_train, 2));

[D10K_validate, labels10K] = generateData(10000, priorL0, priorL1, m01, m02, m11, m12, C, w01, w02, w11, w12);
fprintf('D10K_validate: %d samples generated\n', size(D10K_validate, 2));

figure('Position', [100 100 800 600]);
plotDataset(D50_train, labels50, '50 Training Samples');

figure('Position', [100 100 800 600]);
plotDataset(D500_train, labels500, '500 Training Samples');

figure('Position', [100 100 800 600]);
plotDataset(D5000_train, labels5000, '5000 Training Samples');

figure('Position', [100 100 800 600]);
plotDataset(D10K_validate, labels10K, '10000 Validation Samples');

function [X, labels] = generateData(N, p0, p1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    labels = (rand(N,1) > p0);
    X = zeros(2, N);
    for i = 1:N
        if labels(i) == 0
            if rand() < w01
                X(:,i) = mvnrnd(m01, C)';
            else
                X(:,i) = mvnrnd(m02, C)';
            end
        else
            if rand() < w11
                X(:,i) = mvnrnd(m11, C)';
            else
                X(:,i) = mvnrnd(m12, C)';
            end
        end
    end
end

function plotDataset(X, labels, titleText)
    idx0 = labels == 0;
    idx1 = labels == 1;
    scatter(X(1,idx0), X(2,idx0), 50, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    scatter(X(1,idx1), X(2,idx1), 50, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('x_1', 'FontSize', 12);
    ylabel('x_2', 'FontSize', 12);
    title(titleText, 'FontSize', 14);
    legend('Class 0', 'Class 1', 'Location', 'best');
    grid on;
    axis equal;
    xlim([-4 4]);
    ylim([-4 4]);
end


