function plotData(X, y)
%PLOTDATA Рисует выборку X и y

% Создаем новый холст
figure; hold on;

% ====================== НАЧАЛО ВАШЕГО КОДА ======================

% Разделяем данные на положительные и отрицательные примеры
pos = find(y == 1);
neg = find(y == 0);

% Рисуем положительные и отрицательные примеры
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% ====================== КОНЕЦ ВАШЕГО КОДА ======================



hold off;

end
