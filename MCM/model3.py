# -*- coding: utf-8 -*-
from model2 import ImprovedTourismOptimizer2
from model1 import ImprovedTourismOptimizer1

from scipy.optimize import minimize, dual_annealing
import numpy as np
from scipy.optimize import dual_annealing


class ModelTransferOptimizer:
    def __init__(self, model1, model2, transfer_cost_per_person, normalization_constant):
        self.model1 = model1
        self.model2 = model2
        self.transfer_cost_per_person = transfer_cost_per_person
        self.normalization_constant = normalization_constant
        self.model1_x = [0.08, 0.128, 0.113, 0.159]
        self.model2_x = [0.08, 0.138, 0.119, 0.143]

    def compute_model_objective(self, model, Nt):
        # 使用现有模型计算其目标值
        if model == self.model1:
            model_x = [Nt] + self.model1_x
            return -model.objective(model_x)
        else:
            model_x = [Nt] + self.model2_x
            return -model.objective(model_x)


    def objective(self, x, N1_actual, N2_actual):
        # 分流人数
        N_transfer = x[0]

        # 更新模型1和模型2的人数
        N1_new = N1_actual - N_transfer
        N2_new = N2_actual + N_transfer

        # 确保人数不为负
        if N1_new < 0 or N2_new < 0:
            return float('inf')

        # 计算模型1和模型2的目标值
        model1_objective = self.compute_model_objective(self.model1, N1_new)
        model2_objective = self.compute_model_objective(self.model2, N2_new)

        # 计算总花费
        transfer_cost = N_transfer * self.transfer_cost_per_person

        # 目标函数值
        total_objective = model1_objective + model2_objective - (transfer_cost / self.normalization_constant)
        return -total_objective  # 最大化目标函数

    def optimize_transfer(self, N1_actual, N2_actual):
        def wrapped_objective(x):
            return self.objective(x, N1_actual, N2_actual)

        # 搜索范围
        bounds = [(0, N1_actual)]

        # 使用模拟退火算法寻找全局最优解
        result = dual_annealing(
            wrapped_objective,
            bounds,
            maxiter=1000,
            initial_temp=5230,  # 初始温度，调整以控制搜索范围
            visit=2.7,  # 控制搜索步长的参数
            accept=-5.0,  # 接受准则参数
        )

        if result.success:
            return int(result.x[0]), -result.fun
        else:
            raise ValueError("Global optimization failed.")


if __name__ == "__main__":
    # 初始化模型1和模型2
    model1 = ImprovedTourismOptimizer1()
    model2 = ImprovedTourismOptimizer2()  # 可根据需要调整模型2参数

    # 输入实际人数
    N1_actual = 1670000  # 模型1的实际人数
    N2_actual = 553000  # 模型2的实际人数

    # 输入其他参数
    transfer_cost_per_person = 100  # 每人分流成本
    normalization_constant = 400000000  # 标准化常数

    # 优化分流
    optimizer = ModelTransferOptimizer(model1, model2, transfer_cost_per_person, normalization_constant)
    optimal_transfer, optimal_value = optimizer.optimize_transfer(N1_actual, N2_actual)

    print(f"Optimal number of people to transfer: {optimal_transfer}")
    print(f"Optimal objective function value: {optimal_value}")
