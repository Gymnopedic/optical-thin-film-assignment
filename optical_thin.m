%% MLP 代理模型 - 学习曲线实验 + 膜系快速筛选
% 流程:
%   1. seed 生成 5000 组膜厚+光谱, 固定划分 (4000训练池 + 500验证 + 500测试)
%   2. 从 4000 训练池中依次取 500/1000/2000/4000 组训练 MLP
%   3. 验证集/测试集始终不变, 记录各训练量下的精度
%   4. 用最优模型 (4000组) + design_seed 生成 10000 候选
%   5. 按目标波长 560nm MLP 筛选 Top-10, TMM 精确验证
%   6. 按 TMM 真实反射率降序排列, 确定最终 Top-5 设计
% 日期: 2026-09-29

clear; clc; close all;

%% ==================== 配置参数 ====================

% --- TMM 膜系参数 ---
n_H = 2.30;
n_L = 1.45;
n_inc = 1.0;
n_sub = 1.52;
n_layers = [n_H, n_L, n_H, n_L];
theta_inc = 0;
polarization = 'TE';

% --- 波长范围 ---
lambda_start = 400;
lambda_end   = 800;
lambda_step  = 10;
lambda_vec   = lambda_start:lambda_step:lambda_end;
num_lambda   = length(lambda_vec);

% --- 膜厚范围 ---
d_min = 40;
d_max = 180;
num_layers_count = length(n_layers);

% --- 数据集 ---
num_total_samples = 5000;
num_train_pool    = 4000;
num_val_samples   = 500;
num_test_samples  = 500;

% --- 学习曲线: 依次取不同训练量 ---
train_sizes = [500, 1000, 2000, 4000];

% --- MLP 统一结构 ---
hidden_layers = [128, 128, 64];
activation    = 'relu';
loss_func     = 'mse';

% --- 训练参数 ---
num_epochs    = 500;
batch_size    = 64;
learning_rate = 0.001;
beta1         = 0.9;
beta2         = 0.999;
epsilon       = 1e-8;
patience      = 50;

% --- 随机种子 ---
seed          = 270111;
design_seed   = seed + 1;

% --- 设计阶段参数 ---
num_candidates  = 10000;
top_k           = 10;         % MLP 筛选 Top-10 候选
final_top_k     = 5;          % TMM 验证后确定最终 Top-5 设计
target_lambda   = 560;
target_R_min    = 0.90;
tolerance       = 0.02;

%% ==================== 1. 生成数据集 ====================

fprintf('========================================\n');
fprintf('  MLP 代理模型 - 学习曲线 + 快速筛选\n');
fprintf('========================================\n');
fprintf('seed = %d, design_seed = %d\n', seed, design_seed);
fprintf('目标波长: %d nm\n\n', target_lambda);

fprintf('===== 1. 生成数据集 =====\n');
fprintf('总样本: %d (训练池 %d + 验证 %d + 测试 %d)\n', ...
    num_total_samples, num_train_pool, num_val_samples, num_test_samples);
fprintf('波长: %d-%d nm, 步长 %d nm -> 输出维度 %d\n', ...
    lambda_start, lambda_end, lambda_step, num_lambda);
fprintf('膜厚范围: [%d, %d] nm\n\n', d_min, d_max);

rng(seed);
X_all = d_min + (d_max - d_min) * rand(num_total_samples, num_layers_count);

% 固定划分
X_pool = X_all(1:num_train_pool, :);
X_val  = X_all(num_train_pool+1 : num_train_pool+num_val_samples, :);
X_test = X_all(num_train_pool+num_val_samples+1 : end, :);

% 计算 TMM 反射光谱
fprintf('计算 TMM 反射光谱 (5000 组)...\n');
Y_all = zeros(num_total_samples, num_lambda);
for i = 1:num_total_samples
    d_temp = X_all(i, :);
    Y_all(i, :) = calc_TMM_reflectance(lambda_vec, n_layers, d_temp, ...
                                        n_inc, n_sub, theta_inc, polarization);
    if mod(i, 1000) == 0
        fprintf('  进度: %d / %d\n', i, num_total_samples);
    end
end

Y_pool = Y_all(1:num_train_pool, :);
Y_val  = Y_all(num_train_pool+1 : num_train_pool+num_val_samples, :);
Y_test = Y_all(num_train_pool+num_val_samples+1 : end, :);

% 验证集/测试集归一化参数 (从训练池全量计算, 保持一致)
X_mean = mean(X_pool, 1);
X_std  = std(X_pool, 0, 1) + 1e-8;
Y_mean = mean(Y_pool, 1);
Y_std  = std(Y_pool, 0, 1) + 1e-8;

X_val_norm  = (X_val - X_mean) ./ X_std;
X_test_norm = (X_test - X_mean) ./ X_std;
Y_val_norm  = (Y_val - Y_mean) ./ Y_std;
Y_test_norm = (Y_test - Y_mean) ./ Y_std;

%% ==================== 2. 学习曲线实验 ====================

fprintf('\n===== 2. 学习曲线实验 =====\n');
fprintf('训练量: ');
for ts = 1:length(train_sizes)
    fprintf('%d ', train_sizes(ts));
end
fprintf('\n');
fprintf('统一结构: %d', num_layers_count);
for h = 1:length(hidden_layers)
    fprintf(' -> %d', hidden_layers(h));
end
fprintf(' -> %d, 损失: MSE\n\n', num_lambda);

layer_sizes = [num_layers_count, hidden_layers, num_lambda];
num_weights_layers = length(layer_sizes) - 1;

% 存储各训练量的结果
lc_results = struct();
lc_results.train_sizes   = train_sizes;
lc_results.val_mae       = zeros(length(train_sizes), 1);
lc_results.val_rmse      = zeros(length(train_sizes), 1);
lc_results.val_r2        = zeros(length(train_sizes), 1);
lc_results.test_mae      = zeros(length(train_sizes), 1);
lc_results.test_rmse     = zeros(length(train_sizes), 1);
lc_results.test_r2       = zeros(length(train_sizes), 1);
lc_results.best_val_loss = zeros(length(train_sizes), 1);
lc_results.train_time    = zeros(length(train_sizes), 1);
lc_results.final_epoch   = zeros(length(train_sizes), 1);
lc_results.weights       = cell(length(train_sizes), 1);
lc_results.biases        = cell(length(train_sizes), 1);

for si = 1:length(train_sizes)
    cur_size = train_sizes(si);
    fprintf('--- [%d/%d] 训练量: %d ---\n', si, length(train_sizes), cur_size);

    % 从训练池取前 cur_size 组 (固定子集)
    X_tr = X_pool(1:cur_size, :);
    Y_tr = Y_pool(1:cur_size, :);

    X_tr_norm = (X_tr - X_mean) ./ X_std;
    Y_tr_norm = (Y_tr - Y_mean) ./ Y_std;

    % 用 seed 初始化权重 (每次实验相同初始条件)
    rng(seed);
    weights = cell(num_weights_layers, 1);
    biases  = cell(num_weights_layers, 1);
    for l = 1:num_weights_layers
        fan_in  = layer_sizes(l);
        fan_out = layer_sizes(l+1);
        std_val = sqrt(2 / fan_in);
        weights{l} = std_val * randn(fan_in, fan_out);
        biases{l}  = zeros(1, fan_out);
    end

    m_w = cell(num_weights_layers, 1);  v_w = cell(num_weights_layers, 1);
    m_b = cell(num_weights_layers, 1);  v_b = cell(num_weights_layers, 1);
    for l = 1:num_weights_layers
        m_w{l} = zeros(size(weights{l}));  v_w{l} = zeros(size(weights{l}));
        m_b{l} = zeros(size(biases{l}));   v_b{l} = zeros(size(biases{l}));
    end

    % 训练循环
    best_val_loss = inf;
    best_weights  = weights;
    best_biases   = biases;
    no_improve = 0;
    final_epoch = num_epochs;

    train_loss_history = zeros(num_epochs, 1);
    val_loss_history   = zeros(num_epochs, 1);

    tic;
    for epoch = 1:num_epochs
        perm = randperm(cur_size);
        epoch_loss = 0;
        num_batches = ceil(cur_size / batch_size);

        for b = 1:num_batches
            idx_s = (b-1) * batch_size + 1;
            idx_e = min(b * batch_size, cur_size);
            batch_idx = perm(idx_s:idx_e);
            bs = length(batch_idx);

            X_batch = X_tr_norm(batch_idx, :)';
            Y_batch = Y_tr_norm(batch_idx, :)';

            [activations, pre_activations] = forward_pass(X_batch, weights, biases, activation);
            [grad_w, grad_b] = backward_pass(activations, pre_activations, Y_batch, weights, activation);

            for l = 1:num_weights_layers
                m_w{l} = beta1 * m_w{l} + (1 - beta1) * grad_w{l};
                v_w{l} = beta2 * v_w{l} + (1 - beta2) * (grad_w{l} .^ 2);
                m_b{l} = beta1 * m_b{l} + (1 - beta1) * grad_b{l};
                v_b{l} = beta2 * v_b{l} + (1 - beta2) * (grad_b{l} .^ 2);

                m_w_hat = m_w{l} / (1 - beta1^epoch);
                v_w_hat = v_w{l} / (1 - beta2^epoch);
                m_b_hat = m_b{l} / (1 - beta1^epoch);
                v_b_hat = v_b{l} / (1 - beta2^epoch);

                weights{l} = weights{l} - learning_rate * m_w_hat ./ (sqrt(v_w_hat) + epsilon);
                biases{l}  = biases{l}  - learning_rate * m_b_hat ./ (sqrt(v_b_hat) + epsilon);
            end

            Y_pred_norm = activations{end};
            batch_loss = mean(sum((Y_pred_norm - Y_batch) .^ 2, 1));
            epoch_loss = epoch_loss + batch_loss * bs;
        end

        % 验证集评估
        Y_vp = forward_pass_predict(X_val_norm', weights, biases, activation);
        cur_val_loss = mean(sum((Y_vp - Y_val_norm') .^ 2, 1));

        train_loss_history(epoch) = epoch_loss / cur_size;
        val_loss_history(epoch)   = cur_val_loss;

        if cur_val_loss < best_val_loss
            best_val_loss = cur_val_loss;
            best_weights  = weights;
            best_biases   = biases;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        if mod(epoch, 100) == 0 || epoch == 1
            fprintf('  Epoch %4d | Val Loss: %.6f | Best: %.6f\n', ...
                epoch, cur_val_loss, best_val_loss);
        end

        if no_improve >= patience
            final_epoch = epoch;
            fprintf('  Early stopping at epoch %d\n', epoch);
            break;
        end
    end
    lc_results.train_time(si) = toc;
    lc_results.final_epoch(si) = final_epoch;
    lc_results.best_val_loss(si) = best_val_loss;
    lc_results.train_loss_history{si} = train_loss_history(1:final_epoch);
    lc_results.val_loss_history{si}   = val_loss_history(1:final_epoch);

    % 用最优权重评估
    weights = best_weights;
    biases  = best_biases;
    lc_results.weights{si} = weights;
    lc_results.biases{si}  = biases;

    % 验证集指标
    Y_vp = forward_pass_predict(X_val_norm', weights, biases, activation);
    Y_vp_denorm = Y_vp' .* Y_std + Y_mean;
    lc_results.val_mae(si)  = mean(abs(Y_vp_denorm - Y_val), 'all');
    lc_results.val_rmse(si) = sqrt(mean((Y_vp_denorm - Y_val) .^ 2, 'all'));
    lc_results.val_r2(si)   = 1 - sum((Y_val(:) - Y_vp_denorm(:)) .^ 2) / sum((Y_val(:) - mean(Y_val(:))) .^ 2);

    % 测试集指标
    Y_tp = forward_pass_predict(X_test_norm', weights, biases, activation);
    Y_tp_denorm = Y_tp' .* Y_std + Y_mean;
    lc_results.test_mae(si)  = mean(abs(Y_tp_denorm - Y_test), 'all');
    lc_results.test_rmse(si) = sqrt(mean((Y_tp_denorm - Y_test) .^ 2, 'all'));
    lc_results.test_r2(si)   = 1 - sum((Y_test(:) - Y_tp_denorm(:)) .^ 2) / sum((Y_test(:) - mean(Y_test(:))) .^ 2);

    fprintf('  结果: Val MAE=%.4f%% RMSE=%.4f%% R²=%.6f | Test MAE=%.4f%% RMSE=%.4f%% R²=%.6f\n\n', ...
        lc_results.val_mae(si)*100, lc_results.val_rmse(si)*100, lc_results.val_r2(si), ...
        lc_results.test_mae(si)*100, lc_results.test_rmse(si)*100, lc_results.test_r2(si));
end

% 用最大训练量的模型作为最优模型
best_idx = length(train_sizes);
best_model_weights = lc_results.weights{best_idx};
best_model_biases  = lc_results.biases{best_idx};

fprintf('===== 学习曲线汇总 =====\n');
fprintf('%8s | %10s | %10s | %10s | %10s | %10s | %10s\n', ...
    'N_train', 'Val MAE%', 'Val RMSE%', 'Val R²', 'Test MAE%', 'Test RMSE%', 'Test R²');
fprintf('%s\n', repmat('-', 1, 85));
for si = 1:length(train_sizes)
    fprintf('%8d | %10.4f | %10.4f | %10.6f | %10.4f | %10.4f | %10.6f\n', ...
        train_sizes(si), ...
        lc_results.val_mae(si)*100, lc_results.val_rmse(si)*100, lc_results.val_r2(si), ...
        lc_results.test_mae(si)*100, lc_results.test_rmse(si)*100, lc_results.test_r2(si));
end

%% ==================== 3. 测试集光谱对比 (最优模型) ====================

fprintf('\n===== 3. 测试集评估 (最优模型, N=%d) =====\n', train_sizes(best_idx));

Y_test_pred_norm = forward_pass_predict(X_test_norm', best_model_weights, best_model_biases, activation);
Y_test_pred = Y_test_pred_norm' .* Y_std + Y_mean;

mae_final  = lc_results.test_mae(best_idx);
rmse_final = lc_results.test_rmse(best_idx);
r2_final   = lc_results.test_r2(best_idx);

fprintf('MAE:  %.6f (%.4f%%)\n', mae_final, mae_final * 100);
fprintf('RMSE: %.6f (%.4f%%)\n', rmse_final, rmse_final * 100);
fprintf('R²:   %.6f\n', r2_final);

%% ==================== 4. MLP 快速筛选候选膜系 ====================

fprintf('\n===== 4. MLP 快速筛选候选膜系 =====\n');
fprintf('目标: 波长 %d nm, 反射率 >= %.0f%%\n', target_lambda, target_R_min * 100);
fprintf('候选数量: %d (design_seed = %d)\n\n', num_candidates, design_seed);

rng(design_seed);
X_candidates = d_min + (d_max - d_min) * rand(num_candidates, num_layers_count);
X_candidates_norm = (X_candidates - X_mean) ./ X_std;

tic;
Y_candidates_norm = forward_pass_predict(X_candidates_norm', best_model_weights, best_model_biases, activation);
Y_candidates = Y_candidates_norm' .* Y_std + Y_mean;
mlp_screen_time = toc;
fprintf('MLP 筛选 %d 个候选耗时: %.3f 秒 (%.0f 候选/秒)\n', ...
    num_candidates, mlp_screen_time, num_candidates / mlp_screen_time);

[~, lambda_idx] = min(abs(lambda_vec - target_lambda));

R_at_target = Y_candidates(:, lambda_idx);
[~, sort_idx] = sort(R_at_target, 'descend');

qualified_idx = sort_idx(R_at_target(sort_idx) >= target_R_min);
num_qualified = length(qualified_idx);
fprintf('满足 R >= %.0f%% 的候选: %d / %d\n', target_R_min * 100, num_qualified, num_candidates);

top_k_actual = min(top_k, max(num_qualified, 1));
if num_qualified >= top_k
    top_candidates_idx = qualified_idx(1:top_k);
else
    fprintf('警告: 仅 %d 个候选满足阈值, 取反射率最高的 %d 个\n', num_qualified, top_k);
    top_candidates_idx = sort_idx(1:top_k);
    top_k_actual = top_k;
end

fprintf('选出 Top-%d 候选进行 TMM 验证\n\n', top_k_actual);

%% ==================== 5. TMM 精确验证 ====================

fprintf('===== 5. TMM 精确验证 Top-%d 候选 =====\n\n', top_k_actual);

results = zeros(top_k_actual, num_lambda + num_layers_count + 2);

tic;
for i = 1:top_k_actual
    idx = top_candidates_idx(i);
    d_temp = X_candidates(idx, :);

    R_tmm = calc_TMM_reflectance(lambda_vec, n_layers, d_temp, ...
                                  n_inc, n_sub, theta_inc, polarization);
    R_mlp = Y_candidates(idx, :);

    results(i, 1:num_layers_count) = d_temp;
    results(i, num_layers_count+1 : num_layers_count+num_lambda) = R_tmm;
    results(i, end-1) = R_mlp(lambda_idx);
    results(i, end)   = R_tmm(lambda_idx);

    error_abs = abs(R_tmm(lambda_idx) - R_mlp(lambda_idx));
    pass_str = 'PASS';
    if error_abs > tolerance
        pass_str = 'FAIL';
    end

    fprintf('  #%2d | d=[%6.1f, %6.1f, %6.1f, %6.1f] | R_MLP=%.4f | R_TMM=%.4f | err=%.4f | %s\n', ...
        i, d_temp(1), d_temp(2), d_temp(3), d_temp(4), ...
        R_mlp(lambda_idx), R_tmm(lambda_idx), error_abs, pass_str);
end
tmm_verify_time = toc;

fprintf('\nTMM 验证 %d 个候选耗时: %.3f 秒\n', top_k_actual, tmm_verify_time);
fprintf('总筛选耗时 (MLP+TMM): %.3f 秒\n', mlp_screen_time + tmm_verify_time);

R_tmm_targets = results(:, end);
R_mlp_targets = results(:, end-1);
errors = abs(R_tmm_targets - R_mlp_targets);
pass_count = sum(errors <= tolerance);
fprintf('TMM 验证通过率: %d / %d (%.1f%%)\n', ...
    pass_count, top_k_actual, pass_count / top_k_actual * 100);

%% ==================== 6. 确定最终 Top-5 设计 ====================

fprintf('\n===== 6. 按 TMM 真实光谱确定最终 Top-%d 设计 =====\n\n', final_top_k);

% 按 TMM 目标波长反射率降序重新排序
[~, rank_idx] = sort(R_tmm_targets, 'descend');

final_indices = rank_idx(1:final_top_k);
final_results = results(final_indices, :);

fprintf('最终 Top-%d 设计 (按 TMM R@%dnm 降序):\n\n', final_top_k, target_lambda);
fprintf('%4s | %30s | %10s | %10s | %8s\n', ...
    'Rank', '膜厚 [d1,d2,d3,d4] (nm)', 'R_TMM(%)', 'R_MLP(%)', '误差(%)');
fprintf('%s\n', repmat('-', 1, 85));
for i = 1:final_top_k
    orig_idx = final_indices(i);
    d_temp = final_results(i, 1:num_layers_count);
    r_tmm  = final_results(i, end);
    r_mlp  = final_results(i, end-1);
    err_i  = abs(r_tmm - r_mlp);
    fprintf('  %2d | [%5.1f, %5.1f, %5.1f, %5.1f]  | %9.4f | %9.4f | %7.4f\n', ...
        i, d_temp(1), d_temp(2), d_temp(3), d_temp(4), ...
        r_tmm * 100, r_mlp * 100, err_i * 100);
end

fprintf('\n最终 Top-%d 平均 R_TMM@%dnm: %.4f%%\n', ...
    final_top_k, target_lambda, mean(final_results(:, end)) * 100);
fprintf('最终 Top-%d 最低 R_TMM@%dnm: %.4f%%\n', ...
    final_top_k, target_lambda, min(final_results(:, end)) * 100);

%% ==================== 7. 可视化 ====================

% --- 图: 训练集大小对测试误差的影响 ---
figure('Position', [100, 100, 1200, 800]);

subplot(2, 2, 1);
bar(train_sizes, lc_results.test_mae * 100, 0.6, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
hold on;
plot(train_sizes, lc_results.test_mae * 100, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
for k = 1:length(train_sizes)
    text(train_sizes(k), lc_results.test_mae(k) * 100 + 0.3, ...
        sprintf('%.2f%%', lc_results.test_mae(k) * 100), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end
xlabel('训练集大小', 'FontSize', 12);
ylabel('Test MAE (%)', 'FontSize', 12);
title('训练集大小 vs 测试 MAE', 'FontSize', 13);
grid on;
set(gca, 'XTick', train_sizes);

subplot(2, 2, 2);
bar(train_sizes, lc_results.test_rmse * 100, 0.6, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k');
hold on;
plot(train_sizes, lc_results.test_rmse * 100, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
for k = 1:length(train_sizes)
    text(train_sizes(k), lc_results.test_rmse(k) * 100 + 0.3, ...
        sprintf('%.2f%%', lc_results.test_rmse(k) * 100), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end
xlabel('训练集大小', 'FontSize', 12);
ylabel('Test RMSE (%)', 'FontSize', 12);
title('训练集大小 vs 测试 RMSE', 'FontSize', 13);
grid on;
set(gca, 'XTick', train_sizes);

subplot(2, 2, 3);
improvement_mae = zeros(length(train_sizes), 1);
for k = 2:length(train_sizes)
    improvement_mae(k) = (lc_results.test_mae(k-1) - lc_results.test_mae(k)) / lc_results.test_mae(k-1) * 100;
end
bar(train_sizes(2:end), improvement_mae(2:end), 0.5, ...
    'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
for k = 2:length(train_sizes)
    text(train_sizes(k), improvement_mae(k) + 0.5, ...
        sprintf('%.1f%%', improvement_mae(k)), 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end
xlabel('训练集大小', 'FontSize', 12);
ylabel('MAE 改善率 (%)', 'FontSize', 12);
title('增加训练数据对 MAE 的改善率', 'FontSize', 13);
grid on;
set(gca, 'XTick', train_sizes(2:end));

subplot(2, 2, 4);
improvement_rmse = zeros(length(train_sizes), 1);
for k = 2:length(train_sizes)
    improvement_rmse(k) = (lc_results.test_rmse(k-1) - lc_results.test_rmse(k)) / lc_results.test_rmse(k-1) * 100;
end
bar(train_sizes(2:end), improvement_rmse(2:end), 0.5, ...
    'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k');
for k = 2:length(train_sizes)
    text(train_sizes(k), improvement_rmse(k) + 0.5, ...
        sprintf('%.1f%%', improvement_rmse(k)), 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end
xlabel('训练集大小', 'FontSize', 12);
ylabel('RMSE 改善率 (%)', 'FontSize', 12);
title('增加训练数据对 RMSE 的改善率', 'FontSize', 13);
grid on;
set(gca, 'XTick', train_sizes(2:end));

sgtitle(sprintf('训练集大小对测试误差的影响 (seed=%d)', seed), 'FontSize', 14, 'FontWeight', 'bold');

% --- 图: 训练损失 & 验证损失 (各训练量) ---
figure('Position', [100, 100, 1200, 500]);
num_train_sizes = length(train_sizes);
num_cols = min(num_train_sizes, 2);
num_rows = ceil(num_train_sizes / num_cols);
colors_lv = lines(2);
for si = 1:num_train_sizes
    subplot(num_rows, num_cols, si);
    epochs_vec = 1:length(lc_results.train_loss_history{si});
    semilogy(epochs_vec, lc_results.train_loss_history{si}, '-', 'LineWidth', 1.2, ...
        'Color', colors_lv(1,:), 'DisplayName', '训练损失');
    hold on;
    semilogy(epochs_vec, lc_results.val_loss_history{si}, '-', 'LineWidth', 1.2, ...
        'Color', colors_lv(2,:), 'DisplayName', '验证损失');
    xlabel('Epoch', 'FontSize', 11);
    ylabel('MSE Loss (log)', 'FontSize', 11);
    title(sprintf('训练量 N=%d (共 %d epochs)', train_sizes(si), lc_results.final_epoch(si)), 'FontSize', 12);
    legend('FontSize', 9, 'Location', 'northeast');
    grid on;
    hold off;
end
sgtitle(sprintf('MLP 训练损失 & 验证损失曲线 (seed=%d)', seed), 'FontSize', 14, 'FontWeight', 'bold');

% --- 图2: 代表性测试样本 TMM vs MLP 光谱对比 ---
figure('Position', [100, 100, 1200, 700]);

sample_errors = sum((Y_test_pred - Y_test) .^ 2, 2);
[~, idx_best]   = min(sample_errors);
[~, idx_worst]  = max(sample_errors);
sorted_errors   = sort(sample_errors);
median_err_val  = sorted_errors(ceil(end/2));
[~, idx_median] = min(abs(sample_errors - median_err_val));

representative_idx = [idx_best, idx_worst, idx_median];
extra_needed = min(3, num_test_samples - 3);
if extra_needed > 0
    remaining = setdiff(1:num_test_samples, representative_idx);
    extra_idx = remaining(randperm(length(remaining), extra_needed));
    representative_idx = [representative_idx, extra_idx];
end
num_rep = length(representative_idx);

rep_labels = {sprintf('最佳 (MSE=%.4f)', sample_errors(idx_best)), ...
              sprintf('最差 (MSE=%.4f)', sample_errors(idx_worst)), ...
              sprintf('中等 (MSE=%.4f)', sample_errors(idx_median))};
for e = 1:extra_needed
    rep_labels{end+1} = sprintf('随机%d (MSE=%.4f)', e, sample_errors(extra_idx(e)));
end

for k = 1:num_rep
    subplot(2, 4, k);
    idx = representative_idx(k);
    plot(lambda_vec, Y_test(idx, :) * 100, 'b-', 'LineWidth', 1.5, 'DisplayName', 'TMM');
    hold on;
    plot(lambda_vec, Y_test_pred(idx, :) * 100, 'r--', 'LineWidth', 1.5, 'DisplayName', 'MLP');
    fill([lambda_vec, fliplr(lambda_vec)], ...
         [Y_test(idx,:)*100, fliplr(Y_test_pred(idx,:)*100)], ...
         [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', '误差区域');
    xlabel('波长 (nm)', 'FontSize', 10);
    ylabel('反射率 (%)', 'FontSize', 10);
    title(rep_labels{k}, 'FontSize', 10);
    legend('FontSize', 7, 'Location', 'best');
    grid on;
    ylim([0, 100]);
    xlim([lambda_start, lambda_end]);
    hold off;
end

subplot(2, 4, [5 6]);
scatter(sample_errors * 100, (1:num_test_samples)' / num_test_samples * 100, ...
    20, 'filled', 'MarkerEdgeColor', 'none');
xlabel('测试样本 MSE (\times100)', 'FontSize', 11);
ylabel('累积百分位 (%)', 'FontSize', 11);
title('测试集误差分布 (累积)', 'FontSize', 12);
grid on;
xline(sample_errors(idx_best)*100, 'g--', 'LineWidth', 1, 'Label', '最佳');
xline(sample_errors(idx_worst)*100, 'r--', 'LineWidth', 1, 'Label', '最差');

subplot(2, 4, 8);
errors_abs = abs(Y_test_pred - Y_test) * 100;
histogram(errors_abs(:), 50, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'w');
xlabel('全波段平均绝对误差 (%)', 'FontSize', 11);
ylabel('频数', 'FontSize', 11);
title(sprintf('误差直方图 (均值=%.3f%%)', mean(errors_abs(:))), 'FontSize', 12);
grid on;

sgtitle(sprintf('代表性测试样本: TMM 计算光谱 vs MLP 预测光谱 (N=%d)', train_sizes(best_idx)), ...
    'FontSize', 14, 'FontWeight', 'bold');

% --- 图2b: 最大预测误差样本详细分析 ---
figure('Position', [100, 100, 1100, 750]);

idx_maxerr = idx_worst;
spec_tmm_max = Y_test(idx_maxerr, :) * 100;
spec_mlp_max = Y_test_pred(idx_maxerr, :) * 100;
abs_err_max  = abs(spec_tmm_max - spec_mlp_max);
rel_err_max  = abs_err_max ./ (spec_tmm_max + 1e-10) * 100;
d_maxerr     = X_test(idx_maxerr, :);

subplot(2, 2, 1);
plot(lambda_vec, spec_tmm_max, 'b-', 'LineWidth', 2, 'DisplayName', 'TMM');
hold on;
plot(lambda_vec, spec_mlp_max, 'r--', 'LineWidth', 2, 'DisplayName', 'MLP 预测');
fill([lambda_vec, fliplr(lambda_vec)], ...
     [spec_tmm_max, fliplr(spec_mlp_max)], ...
     [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', '误差区域');
xlabel('波长 (nm)', 'FontSize', 11);
ylabel('反射率 (%)', 'FontSize', 11);
title(sprintf('最大误差样本光谱对比 (MSE=%.4f)', sample_errors(idx_maxerr)), 'FontSize', 12);
legend('FontSize', 9, 'Location', 'best');
grid on;
ylim([0, 100]);
xlim([lambda_start, lambda_end]);
hold off;

subplot(2, 2, 2);
bar_lambda = lambda_vec;
bar_err    = abs_err_max;
h_bar = bar(bar_lambda, bar_err, 'FaceColor', [0.85 0.3 0.3], 'EdgeColor', 'none');
hold on;
plot(bar_lambda, movmean(bar_err, 5), 'k-', 'LineWidth', 1.5, 'DisplayName', '滑动均值');
yline(mean(bar_err), 'b--', 'LineWidth', 1.2, 'DisplayName', sprintf('平均误差 %.2f%%', mean(bar_err)));
xlabel('波长 (nm)', 'FontSize', 11);
ylabel('绝对误差 (%)', 'FontSize', 11);
title('逐波段绝对误差分布', 'FontSize', 12);
legend('FontSize', 9, 'Location', 'northeast');
grid on;
xlim([lambda_start, lambda_end]);
hold off;

subplot(2, 2, 3);
plot(lambda_vec, rel_err_max, 'm-', 'LineWidth', 1.5);
hold on;
yline(mean(rel_err_max), 'b--', 'LineWidth', 1.2, 'DisplayName', sprintf('平均相对误差 %.1f%%', mean(rel_err_max)));
xlabel('波长 (nm)', 'FontSize', 11);
ylabel('相对误差 (%)', 'FontSize', 11);
title('逐波段相对误差 (|TMM-MLP|/TMM)', 'FontSize', 12);
legend('FontSize', 9, 'Location', 'northeast');
grid on;
xlim([lambda_start, lambda_end]);
ylim([0, min(100, max(rel_err_max)*1.2)]);
hold off;

subplot(2, 2, 4);
layer_names = {'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'};
b_thick = bar(1:num_layers_count, d_maxerr, 0.5, 'FaceColor', 'flat');
cmap_bar = zeros(num_layers_count, 3);
for j = 1:num_layers_count
    cmap_bar(j, :) = [0.3 0.5+j*0.1 0.9-j*0.1];
end
b_thick.CData = cmap_bar;
set(gca, 'XTickLabel', layer_names);
xlabel('膜层', 'FontSize', 11);
ylabel('厚度 (nm)', 'FontSize', 11);
title(sprintf('样本膜厚输入: [%.1f, %.1f, %.1f, %.1f] nm', d_maxerr), 'FontSize', 11);
grid on;
ylim([0, max(d_maxerr)*1.3]);
for j = 1:num_layers_count
    text(j, d_maxerr(j)+2, sprintf('%.1f', d_maxerr(j)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

sgtitle(sprintf('MLP 最大预测误差样本分析 (测试样本 #%d, N=%d)', idx_maxerr, train_sizes(best_idx)), ...
    'FontSize', 14, 'FontWeight', 'bold');

% --- 图3: Top-10 TMM 验证光谱 ---
figure('Position', [100, 100, 1000, 700]);
colors_top = lines(top_k_actual);
hold on;
for i = 1:top_k_actual
    d_temp = results(i, 1:num_layers_count);
    R_tmm = results(i, num_layers_count+1 : num_layers_count+num_lambda);
    plot(lambda_vec, R_tmm * 100, '-', 'LineWidth', 1.2, 'Color', colors_top(i,:), ...
        'DisplayName', sprintf('#%d [%d,%d,%d,%d]', i, round(d_temp)));
end
xline(target_lambda, 'k--', 'LineWidth', 1.5, 'Label', sprintf('目标 %dnm', target_lambda));
yline(target_R_min * 100, 'g--', 'LineWidth', 1.5, 'Label', sprintf('阈值 %.0f%%', target_R_min*100));
xlabel('波长 (nm)', 'FontSize', 12);
ylabel('反射率 (%)', 'FontSize', 12);
title(sprintf('MLP 筛选 Top-%d 候选 TMM 验证光谱', top_k_actual), 'FontSize', 13);
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
xlim([lambda_start, lambda_end]);
ylim([0, 100]);
hold off;

% --- 图4: 最终 Top-5 设计 TMM 光谱 ---
figure('Position', [100, 100, 900, 550]);
colors_final = lines(final_top_k);
hold on;
for i = 1:final_top_k
    d_temp = final_results(i, 1:num_layers_count);
    R_tmm_f = final_results(i, num_layers_count+1 : num_layers_count+num_lambda);
    plot(lambda_vec, R_tmm_f * 100, '-', 'LineWidth', 2, 'Color', colors_final(i,:), ...
        'DisplayName', sprintf('Top-%d [%d,%d,%d,%d] R=%.2f%%', ...
        i, round(d_temp), final_results(i,end)*100));
end
xline(target_lambda, 'k--', 'LineWidth', 1.5, 'Label', sprintf('目标 %dnm', target_lambda));
yline(target_R_min * 100, 'g--', 'LineWidth', 1.5, 'Label', sprintf('阈值 %.0f%%', target_R_min*100));
xlabel('波长 (nm)', 'FontSize', 12);
ylabel('反射率 (%)', 'FontSize', 12);
title(sprintf('最终 Top-%d 设计 TMM 反射光谱 (按 R@%dnm 降序)', final_top_k, target_lambda), 'FontSize', 13);
legend('Location', 'eastoutside', 'FontSize', 9);
grid on;
xlim([lambda_start, lambda_end]);
ylim([0, 100]);
hold off;

% --- 图4b: MLP 预测光谱 vs TMM 验证光谱 (含目标波长标注) ---
figure('Position', [100, 100, 1000, 700]);
colors_cmp = lines(top_k_actual);
hold on;
for i = 1:top_k_actual
    idx = top_candidates_idx(i);
    R_tmm_cmp = results(i, num_layers_count+1 : num_layers_count+num_lambda);
    R_mlp_cmp = Y_candidates(idx, :);
    d_temp = results(i, 1:num_layers_count);
    plot(lambda_vec, R_tmm_cmp * 100, '-', 'LineWidth', 1.8, 'Color', colors_cmp(i,:), ...
        'DisplayName', sprintf('#%d TMM [%d,%d,%d,%d]', i, round(d_temp)));
    plot(lambda_vec, R_mlp_cmp * 100, '--', 'LineWidth', 1.2, 'Color', colors_cmp(i,:), ...
        'DisplayName', sprintf('#%d MLP', i));
end
xline(target_lambda, 'k-', 'LineWidth', 2, 'Label', sprintf('目标 %dnm', target_lambda), ...
    'LabelOrientation', 'horizontal', 'FontSize', 10);
yline(target_R_min * 100, 'g--', 'LineWidth', 1.5, 'Label', sprintf('阈值 %.0f%%', target_R_min*100));
xlabel('波长 (nm)', 'FontSize', 12);
ylabel('反射率 (%)', 'FontSize', 12);
title(sprintf('MLP 预测光谱 vs TMM 验证光谱 (Top-%d 候选, 目标 %dnm)', top_k_actual, target_lambda), ...
    'FontSize', 13);
legend('Location', 'eastoutside', 'FontSize', 7, 'NumColumns', 2);
grid on;
xlim([lambda_start, lambda_end]);
ylim([0, 100]);
hold off;

% --- 图5b: Top-5 设计参数表格 ---
fig_tbl = figure('Position', [100, 100, 820, 320], 'Name', 'Top-5 设计参数表');
t_colnames = {'排名', 'd1 (nm)', 'd2 (nm)', 'd3 (nm)', 'd4 (nm)', ...
              sprintf('R_TMM@%dnm (%%)', target_lambda), ...
              sprintf('R_MLP@%dnm (%%)', target_lambda), ...
              '误差 (%)'};
t_data = cell(final_top_k, 8);
for i = 1:final_top_k
    d_temp = final_results(i, 1:num_layers_count);
    r_tmm  = final_results(i, end) * 100;
    r_mlp  = final_results(i, end-1) * 100;
    err_i  = abs(r_tmm - r_mlp);
    t_data(i, :) = {i, d_temp(1), d_temp(2), d_temp(3), d_temp(4), ...
                    sprintf('%.2f', r_tmm), sprintf('%.2f', r_mlp), sprintf('%.3f', err_i)};
end
t = uitable(fig_tbl, 'Data', t_data, 'ColumnName', t_colnames, ...
    'Units', 'normalized', 'Position', [0.02, 0.05, 0.96, 0.82], ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'ColumnWidth', {40, 70, 70, 70, 70, 110, 110, 80});

% --- 图6: 筛选流程示意图 ---
figure('Position', [100, 100, 1100, 320]);
ax_f = axes('Position', [0.02, 0.15, 0.96, 0.7]);
axis off;

n_boxes = 6;
box_w = 0.11;
box_h = 0.65;
y_bot = 0.15;
x_centers = linspace(0.08, 0.92, n_boxes);
arrow_gap = 0.015;

boxes_title = {sprintf('%d 随机候选', num_candidates), 'MLP 快速预测', ...
               sprintf('Top-%d 筛选', top_k_actual), 'TMM 精确验证', ...
               sprintf('TMM 排序'), sprintf('最终 Top-%d 设计', final_top_k)};
boxes_sub   = {'(膜厚 40-180nm)', ...
               sprintf('(%.1f ms/候选)', mlp_screen_time/num_candidates*1000), ...
               sprintf('R@%dnm >= %.0f%%', target_lambda, target_R_min*100), ...
               sprintf('(%.1f ms/候选)', tmm_verify_time/top_k_actual*1000), ...
               sprintf('按 R_TMM@%dnm 降序', target_lambda), ...
               sprintf('平均 R=%.2f%%', mean(final_results(:,end))*100)};
box_colors = [0.7 0.85 1; 0.6 0.9 0.6; 1 0.9 0.6; 1 0.7 0.7; 0.9 0.8 1; 0.7 1 0.7];

for i = 1:n_boxes
    rectangle('Position', [x_centers(i)-box_w/2, y_bot, box_w, box_h], ...
        'Curvature', 0.15, 'FaceColor', box_colors(i,:), 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(x_centers(i), y_bot+box_h*0.62, boxes_title{i}, 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'FontWeight', 'bold');
    text(x_centers(i), y_bot+box_h*0.30, boxes_sub{i}, 'HorizontalAlignment', 'center', 'FontSize', 8);
    if i < n_boxes
        arr_x1 = x_centers(i) + box_w/2 + arrow_gap;
        arr_x2 = x_centers(i+1) - box_w/2 - arrow_gap;
        arr_y  = y_bot + box_h/2;
        annotation('arrow', [arr_x1, arr_x2], [arr_y, arr_y], ...
            'LineWidth', 2, 'Color', [0.25 0.25 0.25], ...
            'HeadWidth', 12, 'HeadLength', 8);
    end
end
title('MLP 快速筛选 + TMM 验证流程', 'FontSize', 13, 'FontWeight', 'bold');

%% ==================== 8. 导出数据 ====================

% 导出学习曲线
fid_lc = fopen('learning_curve_results.csv', 'w');
fprintf(fid_lc, 'N_train,Val_MAE(%),Val_RMSE(%),Val_R2,Test_MAE(%),Test_RMSE(%),Test_R2,Best_Val_Loss,Train_Time(s),Epochs\n');
for si = 1:length(train_sizes)
    fprintf(fid_lc, '%d,%.6f,%.6f,%.8f,%.6f,%.6f,%.8f,%.8f,%.2f,%d\n', ...
        train_sizes(si), ...
        lc_results.val_mae(si)*100, lc_results.val_rmse(si)*100, lc_results.val_r2(si), ...
        lc_results.test_mae(si)*100, lc_results.test_rmse(si)*100, lc_results.test_r2(si), ...
        lc_results.best_val_loss(si), lc_results.train_time(si), lc_results.final_epoch(si));
end
fclose(fid_lc);
fprintf('已导出: learning_curve_results.csv\n');

% 导出 Top-10 验证结果
fid = fopen('TMM_verified_candidates.csv', 'w');
fprintf(fid, 'Rank,d_H1(nm),d_L1(nm),d_H2(nm),d_L2(nm)');
for k = 1:num_lambda
    fprintf(fid, ',R_TMM_%.0fnm(%%)', lambda_vec(k));
end
fprintf(fid, ',R_MLP@%dnm(%%),R_TMM@%dnm(%%),Error(%%)\n', target_lambda, target_lambda);
for i = 1:top_k_actual
    fprintf(fid, '%d', i);
    for j = 1:num_layers_count
        fprintf(fid, ',%.2f', results(i, j));
    end
    for k = 1:num_lambda
        fprintf(fid, ',%.6f', results(i, num_layers_count+k) * 100);
    end
    fprintf(fid, ',%.6f,%.6f,%.6f\n', ...
        results(i, end-1)*100, results(i, end)*100, errors(i)*100);
end
fclose(fid);
fprintf('已导出: TMM_verified_candidates.csv\n');

% 导出最终 Top-5 设计
fid5 = fopen('final_top5_designs.csv', 'w');
fprintf(fid5, 'Final_Rank,d_H1(nm),d_L1(nm),d_H2(nm),d_L2(nm)');
for k = 1:num_lambda
    fprintf(fid5, ',R_TMM_%.0fnm(%%)', lambda_vec(k));
end
fprintf(fid5, ',R_MLP@%dnm(%%),R_TMM@%dnm(%%),Error(%%)\n', target_lambda, target_lambda);
for i = 1:final_top_k
    fprintf(fid5, '%d', i);
    for j = 1:num_layers_count
        fprintf(fid5, ',%.2f', final_results(i, j));
    end
    for k = 1:num_lambda
        fprintf(fid5, ',%.6f', final_results(i, num_layers_count+k) * 100);
    end
    fprintf(fid5, ',%.6f,%.6f,%.6f\n', ...
        final_results(i, end-1)*100, final_results(i, end)*100, ...
        abs(final_results(i,end) - final_results(i,end-1))*100);
end
fclose(fid5);
fprintf('已导出: final_top5_designs.csv\n');

% 保存模型 + 学习曲线
model_data = struct();
model_data.weights = best_model_weights;
model_data.biases  = best_model_biases;
model_data.activation = activation;
model_data.X_mean = X_mean;
model_data.X_std  = X_std;
model_data.Y_mean = Y_mean;
model_data.Y_std  = Y_std;
model_data.lambda_vec = lambda_vec;
model_data.n_layers = n_layers;
model_data.n_inc = n_inc;
model_data.n_sub = n_sub;
model_data.d_min = d_min;
model_data.d_max = d_max;
model_data.seed = seed;
model_data.design_seed = design_seed;
model_data.target_lambda = target_lambda;
model_data.train_sizes = train_sizes;
model_data.lc_results = lc_results;
model_data.results = results;
model_data.top_k_actual = top_k_actual;
model_data.final_results = final_results;
model_data.final_top_k = final_top_k;
model_data.mae = mae_final;
model_data.rmse = rmse_final;
model_data.r2 = r2_final;

save('mlp_tmm_surrogate_model.mat', 'model_data');
fprintf('已保存: mlp_tmm_surrogate_model.mat\n');

%% ==================== 核心函数 ====================

function [activations, pre_activations] = forward_pass(X, weights, biases, activation)
    num_layers = length(weights);
    activations = cell(num_layers + 1, 1);
    pre_activations = cell(num_layers, 1);
    activations{1} = X;
    for l = 1:num_layers
        Z = weights{l}' * activations{l} + biases{l}';
        pre_activations{l} = Z;
        if l < num_layers
            A = apply_activation(Z, activation);
        else
            A = Z;
        end
        activations{l+1} = A;
    end
end

function A_out = forward_pass_predict(X, weights, biases, activation)
    num_layers = length(weights);
    A = X;
    for l = 1:num_layers
        Z = weights{l}' * A + biases{l}';
        if l < num_layers
            A = apply_activation(Z, activation);
        else
            A = Z;
        end
    end
    A_out = A;
end

function A = apply_activation(Z, activation)
    switch lower(activation)
        case 'relu',       A = max(0, Z);
        case 'tanh',       A = tanh(Z);
        case 'sigmoid',    A = 1 ./ (1 + exp(-Z));
        case 'leaky_relu', A = max(0.01 * Z, Z);
        otherwise, error('Unknown activation: %s', activation);
    end
end

function dA = activation_derivative(Z, A, activation)
    switch lower(activation)
        case 'relu',       dA = double(Z > 0);
        case 'tanh',       dA = 1 - A .^ 2;
        case 'sigmoid',    dA = A .* (1 - A);
        case 'leaky_relu', dA = double(Z > 0) + 0.01 * double(Z <= 0);
        otherwise, error('Unknown activation: %s', activation);
    end
end

function [grad_w, grad_b] = backward_pass(activations, pre_activations, Y_true, weights, activation)
    num_layers = length(weights);
    batch_size = size(Y_true, 2);
    grad_w = cell(num_layers, 1);
    grad_b = cell(num_layers, 1);
    delta = (activations{end} - Y_true) * (2 / batch_size);
    for l = num_layers:-1:1
        grad_w{l} = activations{l} * delta';
        grad_b{l} = sum(delta, 2)';
        if l > 1
            delta = (weights{l} * delta) .* activation_derivative(...
                pre_activations{l-1}, activations{l}, activation);
        end
    end
end

function R = calc_TMM_reflectance(lambda_vec, n_layers, d_layers, ...
                                   n_inc, n_sub, theta_inc, pol)
    num_lambda = length(lambda_vec);
    R = zeros(1, num_lambda);
    num_layers_count = length(n_layers);
    theta_inc_rad = deg2rad(theta_inc);
    for k = 1:num_lambda
        lambda = lambda_vec(k);
        theta = zeros(num_layers_count + 2, 1);
        theta(1) = theta_inc_rad;
        n_all = [n_inc, n_layers(:)', n_sub];
        for j = 2:num_layers_count + 2
            sin_theta_j = n_all(1) * sin(theta(1)) / n_all(j);
            if abs(sin_theta_j) > 1
                theta(j) = pi/2;
            else
                theta(j) = asin(sin_theta_j);
            end
        end
        eta = zeros(num_layers_count + 2, 1);
        for j = 1:num_layers_count + 2
            if strcmpi(pol, 'TE')
                eta(j) = n_all(j) * cos(theta(j));
            else
                eta(j) = n_all(j) / cos(theta(j));
            end
        end
        M = eye(2);
        for j = 1:num_layers_count
            delta_j = 2 * pi / lambda * n_layers(j) * d_layers(j) * cos(theta(j+1));
            Mj = [cos(delta_j), -1i * sin(delta_j) / eta(j+1);
                  -1i * eta(j+1) * sin(delta_j), cos(delta_j)];
            M = M * Mj;
        end
        eta_0 = eta(1);
        eta_s = eta(num_layers_count + 2);
        r = (eta_0 * M(1,1) + eta_0 * eta_s * M(1,2) - M(2,1) - eta_s * M(2,2)) / ...
            (eta_0 * M(1,1) + eta_0 * eta_s * M(1,2) + M(2,1) + eta_s * M(2,2));
        R(k) = abs(r)^2;
    end
end
