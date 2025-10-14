clear all; close all; clc;

load('q2_data.mat');

decisions = zeros(length(labels), 1);

for i = 1:length(labels)
    x = data(i, :)';
    
    likelihood1 = mvnpdf(x', mu1', Sigma1);
    likelihood2 = mvnpdf(x', mu2', Sigma2);
    likelihood3 = mvnpdf(x', mu3', Sigma3);
    likelihood4 = mvnpdf(x', mu4', Sigma4);
    
    posterior1 = likelihood1 * priors(1);
    posterior2 = likelihood2 * priors(2);
    posterior3 = likelihood3 * priors(3);
    posterior4 = likelihood4 * priors(4);
    
    [~, decisions(i)] = max([posterior1, posterior2, posterior3, posterior4]);
end

confusion_matrix = zeros(4, 4);
for i = 1:4
    for j = 1:4
        confusion_matrix(i, j) = sum(labels == j & decisions == i);
    end
end

confusion_matrix_prob = zeros(4, 4);
for j = 1:4
    class_total = sum(labels == j);
    if class_total > 0
        confusion_matrix_prob(:, j) = confusion_matrix(:, j) / class_total;
    end
end

fprintf('Confusion Matrix P(D=i|L=j):\n');
fprintf('        L=1     L=2     L=3     L=4\n');
for i = 1:4
    fprintf('D=%d   %.3f   %.3f   %.3f   %.3f\n', i, ...
            confusion_matrix_prob(i,1), confusion_matrix_prob(i,2), ...
            confusion_matrix_prob(i,3), confusion_matrix_prob(i,4));
end

error_rate = 1 - sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
fprintf('\nMinimum P(error): %.4f\n', error_rate);

save('q2_classification_results.mat', 'decisions', 'confusion_matrix', ...
     'confusion_matrix_prob', 'error_rate');
