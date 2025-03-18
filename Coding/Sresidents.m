% 数据输入
Nt = [569000, 742000, 951000, 1167000, 1670000]';
Sr = [59.6591, 65.5109, 64.0071, 69.6875, 63.6905]';

% 构建二次函数的设计矩阵
X = [Nt.^2 Nt ones(size(Nt))];

% 使用最小二乘法求解
coeffs = X \ Sr;

% 提取系数
a1 = coeffs(1)  % 二次项系数
a2 = coeffs(2)  % 一次项系数
b = coeffs(3)   % 常数项

% 计算拟合优度 R²
predictions = a1*Nt.^2 + a2*Nt + b;
SSres = sum((Sr - predictions).^2);
SStot = sum((Sr - mean(Sr)).^2);
R2 = 1 - SSres/SStot

% 绘制拟合结果
figure;
plot(Nt, Sr, 'bo', 'DisplayName', 'Data');
hold on;

% 生成更密集的点以绘制平滑曲线
Nt_fine = linspace(min(Nt), max(Nt), 100)';
predictions_fine = a1*Nt_fine.^2 + a2*Nt_fine + b;
plot(Nt_fine, predictions_fine, 'r-', 'DisplayName', 'Fitted Curve');

xlabel('N_t');
ylabel('S_{residents}');
title('Residents Equation Fitting (Quadratic)');
legend;
grid on;

% 输出方程
fprintf('方程: S_residents = %.2e*Nt^2 + %.2e*Nt + %.4f\n', a1, a2, b);