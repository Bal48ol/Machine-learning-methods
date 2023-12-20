%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Загружаем и визуализируем данные ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;

%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nТестирование lrCostFunction() без регуляризации');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 0;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nСтоимость: %f\n', J);
fprintf('Ожидаемая стоимость: 0.734819\n');
fprintf('Градиенты:\n');
fprintf(' %f \n', grad);
fprintf('Ожидаемые градиенты:\n');
fprintf(' 0.146561\n 0.051442\n 0.124722\n 0.198003\n');
fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;

fprintf('\nТестирование lrCostFunction() с регуляризацией');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nСтоимость: %f\n', J);
fprintf('Ожидаемая стоимость: 2.534819\n');
fprintf('Градиенты:\n');
fprintf(' %f \n', grad);
fprintf('Ожидаемые градиенты:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nОбучение мультиклассовой регрессии...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Нажмитие Enter, чтобы продолжить.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nТочность на обучающей выборке: %f\n', mean(double(pred == y)) * 100);

