function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI ���������� ����������� ����� ��� ������ ���������� theta

% ������������� �������� ����������
m = length(y); % ���������� ��������� ��������
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== ������ ������ ���� ======================
    % Instructions: ��������� ���� ��� ������������ ������ �� �������
    %               ���������� theta.
    %
    % Hint: �� ����� ������� ����� ������� �������� �������� �������
    %       ��������� (computeCost) � ���������.
    %

    % ���������� ���������
    gradient = (1 / m) * X' * (X * theta - y);

    % ���������� ���������� theta
    theta = theta - alpha * gradient;

    % ====================== ����� ������ ���� ======================

    % ��������� �������� J �� ������ ��������
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
