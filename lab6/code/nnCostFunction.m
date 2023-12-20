function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Вычисляет значение функции стоимости двухслойной нейронной сети

% Преобразуем "развернутый" вектор параметров сети nn_params обратно в матрицы весов Theta1 и Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Инициализация полезных переменных
m = size(X, 1);

% Вы должны вернуть корректные значения следующих переменных
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== НАЧАЛО ВАШЕГО КОДА ======================

% Прямое распространение
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Преобразуем y в матрицу бинарных значений
y_matrix = eye(num_labels)(y,:);

% Вычисляем функцию стоимости
J = (1/m) * sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3)));

% Добавляем регуляризацию
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Рассчитываем ошибки на выходном слое (шаг 2)
delta3 = a3 - y_matrix;

% Рассчитываем ошибки на скрытом слое (шаг 3)
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% Рассчитываем градиенты (шаг 4)
Delta1 = delta2' * a1;
Delta2 = delta3' * a2;

% Вычисляем градиенты без регуляризации
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Добавляем регуляризацию к градиентам, исключая смещения (шаг 5)
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta1_grad = Theta1_grad + (lambda/m) * Theta1;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2;

% ====================== КОНЕЦ ВАШЕГО КОДА ======================

% Разворачиваем вектор градиента
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
