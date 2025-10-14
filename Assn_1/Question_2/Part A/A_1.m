clear all; close all; clc;

priors = [0.25, 0.25, 0.25, 0.25];

mu1 = [-2; -2];
mu2 = [2; -2];
mu3 = [2; 2];
mu4 = [-2; 2];

Sigma1 = [0.8, 0.2; 0.2, 0.5];
Sigma2 = [0.5, -0.3; -0.3, 0.8];
Sigma3 = [0.6, 0; 0, 0.6];
Sigma4 = [1.0, 0.4; 0.4, 0.3];

N = 10000;
rng(42);

cumulative_priors = cumsum(priors);
u = rand(N, 1);
labels = zeros(N, 1);

for i = 1:N
    labels(i) = find(u(i) <= cumulative_priors, 1);
end

N1 = sum(labels == 1);
N2 = sum(labels == 2);
N3 = sum(labels == 3);
N4 = sum(labels == 4);

data = zeros(N, 2);
data(labels == 1, :) = mvnrnd(mu1', Sigma1, N1);
data(labels == 2, :) = mvnrnd(mu2', Sigma2, N2);
data(labels == 3, :) = mvnrnd(mu3', Sigma3, N3);
data(labels == 4, :) = mvnrnd(mu4', Sigma4, N4);

fprintf('Samples generated per class:\n');
fprintf('Class 1: %d\nClass 2: %d\nClass 3: %d\nClass 4: %d\n', N1, N2, N3, N4);
fprintf('Total: %d\n', N);

save('q2_data.mat', 'data', 'labels', 'priors', ...
     'mu1', 'mu2', 'mu3', 'mu4', ...
     'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4', ...
     'N1', 'N2', 'N3', 'N4');
