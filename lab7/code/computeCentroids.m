function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS возвращает новые центроиды как среднее значение точек, приписанных к ним


% Полезные переменные
[m n] = size(X);

% Вы должны вернуть корректные значения следующих переменных
centroids = zeros(K, n);


% ====================== НАЧАЛО ВАШЕГО КОДА ======================
% Инструкции: Для каждого центроида вычислите среднее значение от всех точек, отнесенных к нему
%               Более конкретно, вектор столбец centroids(i, :)
%               должен содержать средние значения примером выборки X, отнесенных к i-му центроиду
%
% Замечание: Вы можете использовать цикл for
%
for i=1:K
% Найти все точки, приписанные к i-му центроиду
points = X(idx == i, :);

% Вычисляем среднее значение точек
centroid = mean(points);

% Присваиваем новый центроид
centroids(i, :) = centroid;
end








% ====================== КОНЕЦ ВАШЕГО КОДА ======================


end

