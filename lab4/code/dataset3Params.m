function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS возвращает ваши значения параметров C и sigma для части 3 упражнения

% Вы должны заменить следующие значения на найденные вами оптимальные
C = 1;
sigma = 0.3;

% ====================== НАЧАЛО ВАШЕГО КОДА ======================
% Инструкции: Напишите код, который находит оптимальные значения параметров C и sigma
%               После того как код отработает и найдет нужные значения, вы можете его
%               закомментировать и вставить найденные значения в строки 5 и 6 данного файла
%               Вы можете использовать функцию svmPredict для получения прогноза от SVM
%               на валидационной выборке. Например, вызов
%                   predictions = svmPredict(model, Xval);
%               вернет результаты классификации на валидационной выборке.
%
%  Замечание: Вы можете вычислить ошибку прогноза с помощью формулы
%        mean(double(predictions ~= yval))
%

% Задаем значения C и sigma, которые нужно перебрать
values = [0.01 0.03 0.1 0.3 1 3 10 30];

% Инициализируем переменные для хранения наилучшего значения ошибки и соответствующих параметров
best_error = Inf;

% Перебираем все пары значений C и sigma
for i = 1:length(values)
  for j = 1:length(values)
    % Получаем текущие значения C и sigma
    C_temp = values(i);
    sigma_temp = values(j);
    % Обучаем SVM на тренировочной выборке с текущими значениями C и sigma
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    % Получаем прогноз на валидационной выборке
    predictions = svmPredict(model, Xval);
    % Вычисляем ошибку прогноза
    error = mean(double(predictions ~= yval));
    % Если текущая ошибка меньше наилучшей, обновляем значения
    if error < best_error
      best_error = error;
      C = C_temp;
      sigma = sigma_temp;
    end
  end
end

% ====================== КОНЕЦ ВАШЕГО КОДА ======================

end
