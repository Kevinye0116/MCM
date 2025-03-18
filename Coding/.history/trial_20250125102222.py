# -*- coding: utf-8 -*-
# 原始模型NSGA-II实现（未优化版本）
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools

# ---------------------------
# 问题定义与参数设置
# ---------------------------
# 决策变量范围 (游客量, 人均税收)
N_TOURISTS_MIN = 50_000  # 最小游客量（人/年）
N_TOURISTS_MAX = 200_000  # 最大游客量（人/年）
TAX_PER_TOURIST_MIN = 10  # 最小人均税收（美元）
TAX_PER_TOURIST_MAX = 50  # 最大人均税收（美元）

# 目标函数权重：最大化经济、社会，最小化环境负担
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


# ---------------------------
# 目标函数定义（原始线性模型）
# ---------------------------
def evaluate_original(individual):
    n_tourists, tax_per_tourist = individual

    # === 经济目标 ===
    total_tax = tax_per_tourist * n_tourists
    revenue = 500 * n_tourists  # 基础收入（美元）
    economic = revenue + total_tax

    # === 社会目标 ===
    # 居民满意度（线性递减）
    resident_sat = 80 - 0.0002 * n_tourists
    # 游客满意度（线性递减）
    tourist_sat = 90 - 0.1 * tax_per_tourist
    social = 0.5 * resident_sat + 0.5 * tourist_sat

    # === 环境目标 ===
    co2_emission = 0.05 * n_tourists  # 吨CO2

    return economic, social, co2_emission


# ---------------------------
# 遗传算法配置（原始约束）
# ---------------------------
def setup_toolbox_original():
    toolbox = base.Toolbox()

    # 定义变量和种群
    toolbox.register(
        "attr_n_tourists", np.random.uniform, N_TOURISTS_MIN, N_TOURISTS_MAX
    )
    toolbox.register(
        "attr_tax", np.random.uniform, TAX_PER_TOURIST_MIN, TAX_PER_TOURIST_MAX
    )
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_n_tourists, toolbox.attr_tax),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算子
    toolbox.register("evaluate", evaluate_original)
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[N_TOURISTS_MIN, TAX_PER_TOURIST_MIN],
        up=[N_TOURISTS_MAX, TAX_PER_TOURIST_MAX],
        eta=20.0,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[N_TOURISTS_MIN, TAX_PER_TOURIST_MIN],
        up=[N_TOURISTS_MAX, TAX_PER_TOURIST_MAX],
        eta=20.0,
        indpb=0.1,
    )
    toolbox.register("select", tools.selNSGA2)

    # === 原始约束处理（惩罚函数法）===
    def check_constraints(individual):
        n_tourists, tax = individual
        violations = 0

        # 约束1: 游客量不超过基础设施承载力
        if n_tourists > 180_000:  # 原始模型设定固定上限
            violations += (n_tourists - 180_000) / 1e4

        # 约束2: 居民满意度不低于50
        resident_sat = 80 - 0.0002 * n_tourists
        if resident_sat < 50:
            violations += 50 - resident_sat

        return violations

    # 将约束违反量作为惩罚项加入目标函数
    original_eval = toolbox.evaluate

    def constrained_evaluate(individual):
        penalty = check_constraints(individual)
        obj1, obj2, obj3 = original_eval(individual)
        return (obj1 - 1e4 * penalty, obj2 - 1e4 * penalty, obj3)  # 惩罚项加权

    toolbox.register("evaluate", constrained_evaluate)

    return toolbox


# ---------------------------
# 运行优化
# ---------------------------
def main_original():
    # 参数设置
    population_size = 100
    n_generations = 50
    crossover_prob = 0.7  # 修复交叉概率
    mutation_prob = 0.2

    toolbox = setup_toolbox_original()
    pop = toolbox.population(n=population_size)

    # 统计指标
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # 运行NSGA-II
    result, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=population_size,
        lambda_=population_size,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=n_generations,
        stats=stats,
        verbose=True,
    )

    return result, logbook


# ---------------------------
# 结果可视化（原始模型）
# ---------------------------
def plot_pareto_original(result):
    # 提取非支配解
    fits = np.array([ind.fitness.values for ind in result])
    fronts = tools.sortLogNondominated(result, len(result))
    pareto_front = np.array([ind.fitness.values for ind in fronts[0]])

    # 二维绘图
    plt.figure(figsize=(10, 6))

    # 经济-社会
    plt.subplot(121)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="r", alpha=0.7)
    plt.xlabel("Economic Revenue ($)")
    plt.ylabel("Social Satisfaction")
    plt.title("Economic vs Social")
    plt.grid(True)

    # 经济-环境
    plt.subplot(122)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 2], c="b", alpha=0.7)
    plt.xlabel("Economic Revenue ($)")
    plt.ylabel("CO2 Emissions (ton)")
    plt.title("Economic vs Environmental")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------
# 三维帕累托前沿可视化函数
# ---------------------------
def plot_3d_pareto(result):
    # 提取所有个体的目标值
    fits = np.array([ind.fitness.values for ind in result])
    economic = fits[:, 0]
    social = fits[:, 1]
    environmental = fits[:, 2]

    # 非支配排序获取帕累托前沿
    fronts = tools.sortLogNondominated(result, len(result))
    pareto_front = np.array([ind.fitness.values for ind in fronts[0]])

    # 创建三维坐标系
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制所有解
    ax.scatter(
        economic,
        social,
        environmental,
        c="gray",
        alpha=0.3,
        label="All Solutions",
        s=20,
    )

    # 绘制帕累托前沿
    ax.scatter(
        pareto_front[:, 0],
        pareto_front[:, 1],
        pareto_front[:, 2],
        c=pareto_front[:, 2],
        cmap="viridis_r",
        edgecolor="k",
        linewidth=0.5,
        s=50,
        label="Pareto Front",
    )

    # 设置坐标轴标签
    ax.set_xlabel("Economic Revenue ($)", labelpad=12)
    ax.set_ylabel("Social Satisfaction", labelpad=12)
    ax.set_zlabel("Environmental Burden (CO2)", labelpad=12)

    # 设置视角
    ax.view_init(elev=25, azim=-45)  # 调整视角高度和方位角

    # 添加颜色条
    cbar = fig.colorbar(ax.collections[1], ax=ax, pad=0.1)
    cbar.set_label("CO2 Emission Level", rotation=270, labelpad=15)

    # 添加图例
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 0.9))

    # 设置标题
    plt.title(
        "3D Pareto Front: Economic-Social-Environmental Trade-off", pad=20, fontsize=12
    )

    # 优化布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result_original, logbook_original = main_original()
    # plot_pareto_original(result_original)
    plot_3d_pareto(result_original)
