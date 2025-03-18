# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from mpl_toolkits.mplot3d import Axes3D
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ==========================
# 1. 问题定义与初始化
# ==========================
# 参数设置
N_TOURISTS_MIN = 50_000  # 游客量下限
N_TOURISTS_MAX = 200_000  # 游客量上限
TAX_MIN = 10  # 人均税收下限(美元)
TAX_MAX = 50  # 人均税收上限(美元)
INVEST_ECO_MIN = 0.1  # 环保投资比例下限
INVEST_ECO_MAX = 0.5  # 环保投资比例上限

# 目标函数权重：最大化经济、社会，最小化环境
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


# ==========================
# 2. 目标函数定义
# ==========================
def evaluate(individual):
    print("Individual structure:", individual)  # 调试输出
    n_tourists, tax, invest_eco = individual

    # 经济目标
    revenue = 500 * n_tourists
    total_tax = tax * n_tourists
    eco_invest = total_tax * invest_eco
    economic = revenue + total_tax - eco_invest

    # 社会目标
    resident_sat = 80 / (1 + np.exp(0.00001 * (n_tourists - 180_000)))  # logistic函数
    tourist_sat = 90 - 0.1 * tax - 0.05 * eco_invest
    social = 0.6 * resident_sat + 0.4 * tourist_sat

    # 环境目标
    co2_base = 0.05 * n_tourists
    co2_reduce = 0.8 * eco_invest
    environmental = co2_base - co2_reduce

    return economic, social, environmental


# ==========================
# 3. 算法配置（修正后）
# ==========================
def setup_toolbox():
    toolbox = base.Toolbox()

    # 独立注册每个变量属性
    toolbox.register(
        "attr_n_tourists", np.random.uniform, N_TOURISTS_MIN, N_TOURISTS_MAX
    )
    toolbox.register("attr_tax", np.random.uniform, TAX_MIN, TAX_MAX)
    toolbox.register(
        "attr_invest_eco", np.random.uniform, INVEST_ECO_MIN, INVEST_ECO_MAX
    )

    # 使用initCycle组合三个独立变量
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_n_tourists, toolbox.attr_tax, toolbox.attr_invest_eco),
        n=1,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 遗传算子配置
    low = [N_TOURISTS_MIN, TAX_MIN, INVEST_ECO_MIN]
    up = [N_TOURISTS_MAX, TAX_MAX, INVEST_ECO_MAX]

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=20.0)
    toolbox.register(
        "mutate", tools.mutPolynomialBounded, low=low, up=up, eta=20.0, indpb=0.1
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    return toolbox


# ==========================
# 4. 运行优化算法
# ==========================
def run_optimization():
    toolbox = setup_toolbox()

    # 算法参数
    pop_size = 100
    n_gen = 50
    cx_prob = 0.7  # 交叉概率
    mut_prob = 0.2  # 变异概率

    pop = toolbox.population(n=pop_size)

    # 运行NSGA-II
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    result, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=n_gen,
        stats=stats,
        verbose=True,
    )

    return result, logbook


# ==========================
# 5. 三维帕累托可视化
# ==========================
def plot_3d_pareto(front):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 提取目标值
    economic = [ind[0] for ind in front]
    social = [ind[1] for ind in front]
    environmental = [ind[2] for ind in front]

    # 创建颜色映射
    colors = environmental
    cmap = plt.cm.get_cmap("viridis_r")
    normalize = plt.Normalize(min(colors), max(colors))

    # 绘制三维散点图
    sc = ax.scatter(
        economic,
        social,
        environmental,
        c=colors,
        cmap=cmap,
        norm=normalize,
        s=50,
        edgecolor="k",
        alpha=0.8,
    )

    # 设置坐标轴
    ax.set_xlabel("Economic Revenue ($M)", labelpad=15, fontsize=12)
    ax.set_ylabel("Social Satisfaction", labelpad=15, fontsize=12)
    ax.set_zlabel("CO2 Emissions (ton)", labelpad=15, fontsize=12)

    # 添加颜色条
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Environmental Impact", rotation=270, labelpad=20, fontsize=12)

    # 设置视角
    ax.view_init(elev=25, azim=-45)

    # 添加标题
    plt.title("3D Pareto Front: Sustainable Tourism Optimization", pad=20, fontsize=14)

    plt.tight_layout()
    plt.show()


# ==========================
# 6. 主程序
# ==========================
if __name__ == "__main__":
    # 运行优化算法
    result, logbook = run_optimization()

    # 提取非支配前沿
    fits = np.array([ind.fitness.values for ind in result])
    nds = NonDominatedSorting().do(fits, only_non_dominated_front=True)
    pareto_front = fits[nds]

    # 绘制三维帕累托前沿
    plot_3d_pareto(pareto_front)

    # 打印最优解示例
    print("\nPareto最优解示例:")
    print(f"{'经济收益($M)':<15}{'社会满意度':<15}{'CO2排放(ton)':<15}")
    for sol in pareto_front[:5]:
        print(f"{sol[0]/1e6:8.2f}M     {sol[1]:8.1f}     {sol[2]:8.0f}")
