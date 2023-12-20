function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Находит лучшее пороговое значение (epsilon) для обнаружения аномалий
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== НАЧАЛО ВАШЕГО КОДА ======================
    % Инструкции: Вычислите метрику F1, используя epsilon в качестве порога.
    %               Код в конце файла сравнит значение F1 с наилучшим текущим значением
    %               и сохранит текущее, если оно окажется лучше.
    %               
    % Замечание: Вы можете использовать predictions = (pval < epsilon), чтобы получить бинарный вектор
    %       из нулей и единиц с предсказаниями аномалий

	predictions = (pval < epsilon);

        tp = sum((predictions == 1) & (yval == 1));
        fp = sum((predictions == 1) & (yval == 0));
        fn = sum((predictions == 0) & (yval == 1));

        precision = tp / (tp + fp);
        recall = tp / (tp + fn);

        F1 = 2 * precision * recall / (precision + recall);


    % ====================== КОНЕЦ ВАШЕГО КОДА ======================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
