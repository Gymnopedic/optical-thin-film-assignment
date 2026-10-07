%% 薄膜反射率计算 - 传输矩阵法 (TMM)
% 计算多层介质薄膜的反射率光谱

clear; clc; close all;

%% 参数设置
lambda_start = 400;    % 起始波长 (nm)
lambda_end   = 800;    % 终止波长 (nm)
lambda_step  = 1;      % 波长步长 (nm)
theta_inc    = 0;      % 入射角 (度)
polarization = 's';    % 偏振: 's' 或 'p'

% 薄膜结构 (从入射介质到基底)
% 格式: {折射率, 厚度(nm)}
% 示例: 空气 / SiO2(100nm) / TiO2(80nm) / SiO2(100nm) / 玻璃基底
layers = {
    1.0,    0;     % 入射介质 (空气)
    1.46,  100;    % SiO2
    2.35,   80;    % TiO2
    1.46,  100;    % SiO2
    1.52,    0     % 玻璃基底
};

%% 计算
lambda = lambda_start:lambda_step:lambda_end;
R = zeros(size(lambda));

for i = 1:length(lambda)
    lam = lambda(i);
    M = eye(2);

    for k = 2:length(layers)-1
        n_k = layers{k, 1};
        d_k = layers{k, 2};

        delta = 2 * pi * n_k * d_k * cosd(theta_t(n_k, layers{1,1}, theta_inc)) / lam;

        if strcmp(polarization, 's')
            eta = n_k * cosd(theta_t(n_k, layers{1,1}, theta_inc));
        else
            eta = n_k / cosd(theta_t(n_k, layers{1,1}, theta_inc));
        end

        M_k = [cos(delta), 1i*sin(delta)/eta;
               1i*eta*sin(delta), cos(delta)];
        M = M * M_k;
    end

    % 入射介质和基底的导纳
    if strcmp(polarization, 's')
        eta_0 = layers{1,1} * cosd(theta_inc);
        eta_sub = layers{end,1} * cosd(theta_t(layers{end,1}, layers{1,1}, theta_inc));
    else
        eta_0 = layers{1,1} / cosd(theta_inc);
        eta_sub = layers{end,1} / cosd(theta_t(layers{end,1}, layers{1,1}, theta_inc));
    end

    % 反射系数
    r = (eta_0 * M(1,1) + eta_0 * eta_sub * M(1,2) - M(2,1) - eta_sub * M(2,2)) / ...
        (eta_0 * M(1,1) + eta_0 * eta_sub * M(1,2) + M(2,1) + eta_sub * M(2,2));

    R(i) = abs(r)^2;
end

%% 绘图
figure('Color', 'w');
plot(lambda, R * 100, 'b-', 'LineWidth', 1.5);
xlabel('波长 (nm)', 'FontSize', 12);
ylabel('反射率 (%)', 'FontSize', 12);
title('多层薄膜反射率光谱', 'FontSize', 14);
grid on;
ylim([0 100]);
xlim([lambda_start lambda_end]);

%% 辅助函数: 折射角 (Snell定律)
function theta = theta_t(n_t, n_i, theta_i)
    sin_t = n_i * sind(theta_i) / n_t;
    sin_t = min(sin_t, 1);
    theta = asind(sin_t);
end
