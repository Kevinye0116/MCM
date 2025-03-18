import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ==========================
# 模型参数与变量定义 (需根据实际数据调整)
# ==========================
# 决策变量范围
N_t_min = 50_000  # 最小游客量（人/天）
N_t_max = 200_000  # 最大游客量（人/天）
tau_t_min = 10  # 最小旅游税（美元/人）
tau_t_max = 50  # 最大旅游税（美元/人）

# 固定参数（示例值，需替换为实际数据）
P_t = 300  # 人均消费（美元/人）
CO2p = 0.05  # 人均碳排放（吨/人）
k1 = 0.1  # 环保投资对环境质量的增益系数
k2 = 0.05  # 游客碳排放对环境质量的损耗系数
k3 = 0.02  # 自然恢复系数
k4 = 0.3  # 基础设施投资对承载力的增益系数
k5 = 0.4  # 基础设施投资比例（P_b = k5 * τ_t * N_t）
k6 = 0.3  # 环保投资比例（P_e = k6 * τ_t * N_t）
C_h_coeff = 1e-5  # 隐藏成本系数（C_h = C_h_coeff * N_t^2）
gamma = 0.1  # 游客满意度对下一期游客量的影响系数
delta = 0.05  # 环境质量对游客量的衰减系数
E_min = 50  # 环境质量最低阈值
CO2_max = 10_000  # 最大允许碳排放（吨/天）
tau_t_max_constr = 40  # 税收上限（τ_t ≤ 40）

# ==========================
# 多目标优化问题定义
# ==========================
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def evaluate(individual):
    N_t, tau_t = individual

    # --- 计算中间变量 ---
    P_b = k5 * tau_t * N_t  # 基础设施投资
    P_e = k6 * tau_t * N_t  # 环保投资
    C_h = C_h_coeff * N_t**2  # 隐藏成本
    CO2_total = N_t * CO2p  # 总碳排放

    # --- 目标1: 总收益 R_e ---
    R_e = (tau_t + P_t) * N_t - P_e - P_b - C_h

    # --- 目标2: 社会满意度 S ---
    S_residents = -0.0002 * N_t + 80  # 线性模型示例
    S_tourists = (-0.01 * tau_t - 0.005 * P_t + 95) / (1 + 0.1 * (N_t / 100_000))
    S = S_residents + S_tourists

    # --- 目标3: 环境质量 E（需动态计算，此处简化）---
    # 假设环境质量稳态（dE/dt = 0）
    E = (
        k1 * P_e - k2 * CO2_total + k3 * (-0.5)
    ) / 0.01  # 假设R_nature = -0.5（冰川退缩）

    # --- 约束处理（惩罚函数法）---
    penalty = 0
    # 财务约束：R_e ≥ 0
    if R_e < 0:
        penalty += 1e6 * abs(R_e)
    # 环境约束：E ≥ E_min 和 CO2_total ≤ CO2_max
    if E < E_min:
        penalty += 1e4 * (E_min - E)
    if CO2_total > CO2_max:
        penalty += 1e4 * (CO2_total - CO2_max)
    # 社会约束：S_residents ≥ 60
    if S_residents < 60:
        penalty += 1e4 * (60 - S_residents)

    return (R_e - penalty, S - penalty, -CO2_total - penalty)


# ==========================
# 遗传算法配置
# ==========================
toolbox = base.Toolbox()

# 定义变量和种群
toolbox.register("attr_N_t", np.random.uniform, N_t_min, N_t_max)
toolbox.register("attr_tau_t", np.random.uniform, tau_t_min, tau_t_max)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attr_N_t, toolbox.attr_tau_t),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册遗传算子
toolbox.register("evaluate", evaluate)
toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=[N_t_min, tau_t_min],
    up=[N_t_max, tau_t_max],
    eta=20.0,
)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=[N_t_min, tau_t_min],
    up=[N_t_max, tau_t_max],
    eta=20.0,
    indpb=0.1,
)
toolbox.register("select", tools.selNSGA2)


# ==========================
# 运行优化
# ==========================
# 在运行优化的 main() 函数中调整参数
def main():
    population_size = 100
    n_generations = 50
    crossover_prob = 0.7  # 原为 0.9，现调整为 0.7
    mutation_prob = 0.2  # 保持 0.2

    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # 运行 NSGA-II
    result, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=population_size,
        lambda_=population_size,
        cxpb=crossover_prob,
        mutpb=mutation_prob,  # 传递调整后的参数
        ngen=n_generations,
        stats=stats,
        verbose=True,
    )

    return result, logbook


# ==========================
# 可视化帕累托前沿
# ==========================
def plot_pareto_front(result):
    fits = np.array([ind.fitness.values for ind in result])
    nds = NonDominatedSorting().do(fits, only_non_dominated_front=True)
    pareto_front = fits[nds]

    fig = plt.figure(figsize=(15, 5))

    # 经济 vs 社会满意度
    ax1 = fig.add_subplot(131)
    ax1.scatter(pareto_front[:, 0], pareto_front[:, 1], c="blue")
    ax1.set_xlabel("Total Revenue ($)")
    ax1.set_ylabel("Social Satisfaction")
    ax1.grid(True)

    # 经济 vs 环境负担（CO2）
    ax2 = fig.add_subplot(132)
    ax2.scatter(pareto_front[:, 0], pareto_front[:, 2], c="red")
    ax2.set_xlabel("Total Revenue ($)")
    ax2.set_ylabel("CO2 Emissions (ton)")
    ax2.grid(True)

    # 社会满意度 vs 环境负担
    ax3 = fig.add_subplot(133)
    ax3.scatter(pareto_front[:, 1], pareto_front[:, 2], c="green")
    ax3.set_xlabel("Social Satisfaction")
    ax3.set_ylabel("CO2 Emissions (ton)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result, logbook = main()
    plot_pareto_front(result)
