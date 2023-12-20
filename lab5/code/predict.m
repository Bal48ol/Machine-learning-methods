function p = predict(Theta1, Theta2, X)
%PREDICT Прогнозирует значения классов входных данных с помощью обученной
%   нейронной сети

% Инициализация полезных переменных
m = size(X, 1);
num_labels = size(Theta2, 1);

% Вы должны вернуть корректные значения следующих переменных
p = zeros(size(X, 1), 1);

% ====================== НАЧАЛО ВАШЕГО КОДА ======================
% Подсказка: Почитайте документацию к функции max. Она позволяет искать
%       максимальные значения как по строкам, так и по столбцам матрицы.

% Добавляем столбец единиц к матрице X
X = [ones(m, 1) X];

% Вычисляем активацию для скрытого слоя
z2 = X * Theta1';
a2 = sigmoid(z2);

% Добавляем столбец единиц к матрице a2
a2 = [ones(size(a2, 1), 1) a2];

% Вычисляем активацию для выходного слоя
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Прогнозируем значения
[~, p] = max(a3, [], 2);

% ====================== КОНЕЦ ВАШЕГО КОДА ======================


end
