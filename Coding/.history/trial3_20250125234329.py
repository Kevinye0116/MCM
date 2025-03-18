from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class TourismOptimization:
    def __init__(self):
        # 模型系数
        self.k1 = 0.5  # 环保投资效率系数
        self.k2 = 0.3  # 碳排放影响系数
        self.k3 = 0.2  # 自然恢复系数
        self.k4 = 0.15  # 基础设施发展系数
        self.k5 = 0.20  # 废物处理投资系数
        self.k6 = 0.05  # 水资源投资系数
        self.k7 = 0.05  # 环保投资系数

        # 社会满意度参数
        self.a1 = -1.975284171471327e-11  # 游客数量对居民满意度的影响系数
        self.a2 = 4.772497706879391e-05  # 旅游税对游客满意度的影响系数
        self.a3 = 0.015  # 消费水平对游客满意度的影响系数
        self.b1 = 39.266042633247565  # 居民基础满意度
        self.b2 = 60  # 游客基础满意度

        # 约束条件上限
        self.Nt_max = 5000000  # 最大游客数量
        self.tau_max = 7  # 最大旅游税
        self.Pt = 190  # 固定游客消费为常数
        self.Pt_max = 190  # 更新最大游客消费为固定值
        self.CO2_max = 500000  # 最大碳排放量
        self.E = 0  # 最小环境质量指数

        # 其他参数
        self.CO2p = 0.184  # 人均碳排放量

        # 系统动态参数
        self.dt = 0.1  # 时间步长

        # 新增参数
        self.Cwater = 18900000  # 水资源容量
        self.Cwaste = 2549110  # 废物处理容量
        self.a4 = 0.01  # 游客数量对游客满意度的影响系数
        self.b3 = 10  # 废物处理基准增长率

        # 新增影响系数
        self.k8 = 0.1  # 环保投资对碳排放的影响系数
        self.k9 = 0.4  # 废物处理投资对废物处理容量的影响系数
        self.k10 = 0.1  # 水资源投资对水资源容量的影响系数

        # 初始容量
        self.CO2p_initial = 0.184  # 初始人均碳排放量
        self.Cwaste_initial = 2549110  # 初始废物处理容量
        self.Cwater_initial = 18900000  # 初始水资源容量

    def calculate_revenue(self, x: np.ndarray) -> float:
        """计算总收入"""
        Nt, tau_t = x
        return self.Pt * Nt  # 使用固定的Pt值

    def calculate_social_satisfaction(self, x: np.ndarray) -> float:
        """计算社会满意度，确保满意度在0-100范围内"""
        Nt, tau_t = x

        # 居民满意度包含二次项
        S_residents = self.a1 * Nt * Nt + self.a2 * Nt + self.b1
        # 限制居民满意度在0-100范围内
        S_residents = np.clip(S_residents, 0, 100)

        # 游客满意度
        # S_tourists = self.a3 * self.Pt + self.a4 * Nt + self.b2
        # 限制游客满意度在0-100范围内
        # S_tourists = np.clip(S_tourists, 0, 100)

        # 加权平均
        return S_residents

    def calculate_adjusted_capacities(
        self, x: np.ndarray
    ) -> Tuple[float, float, float]:
        """计算投资后的调整容量"""
        P_waste, P_water, P_e = self.calculate_investments(x)

        # 计算调整后的容量
        CO2p_adjusted = self.CO2p_initial - self.k8 * P_e
        Cwaste_adjusted = self.Cwaste_initial - self.k9 * P_waste
        Cwater_adjusted = self.Cwater_initial - self.k10 * P_water

        # 确保容量不会变为负值
        CO2p_adjusted = max(0, CO2p_adjusted)
        Cwaste_adjusted = max(0, Cwaste_adjusted)
        Cwater_adjusted = max(0, Cwater_adjusted)

        return CO2p_adjusted, Cwaste_adjusted, Cwater_adjusted

    def calculate_environmental_quality(self, x: np.ndarray) -> float:
        """计算环境质量指数"""
        Nt, tau_t = x
        CO2p_adj, Cwaste_adj, Cwater_adj = self.calculate_adjusted_capacities(x)

        return (
            self.k1 * CO2p_adj * Nt / self.CO2_max
            + self.k2 * Nt / Cwaste_adj
            + self.k3 * Nt / Cwater_adj
        )

    # def calculate_dCwaste_dt(self, x: np.ndarray) -> float:
    #     """计算废物处理容量的变化率 dCwaste/dt"""
    #     Nt, tau_t = x
    #     Pb = self.k5 * tau_t * Nt  # 基础设施投资
    #     return self.k4 * Pb + self.b3

    def calculate_investments(self, x: np.ndarray) -> Tuple[float, float, float]:
        """计算三种投资金额"""
        Nt, tau_t = x
        Re = self.calculate_revenue(x)  # 总收入
        P_waste = self.k5 * tau_t * Re / 100  # 废物处理投资
        P_water = self.k6 * tau_t * Re / 100  # 水资源投资
        P_e = self.k7 * tau_t * Re / 100  # 环保投资
        return P_waste, P_water, P_e

    def constraints(self, x: np.ndarray) -> List[float]:
        """检查所有约束条件"""
        Nt, tau_t = x
        CO2p_adj, Cwaste_adj, Cwater_adj = self.calculate_adjusted_capacities(x)

        constraints = [
            self.Nt_max - Nt,  # 游客数量上限约束
            self.tau_max - tau_t,  # 旅游税上限约束
            self.CO2_max - Nt * CO2p_adj,  # 使用调整后的碳排放约束
            self.a1 * Nt * Nt + self.a2 * Nt + self.b1 - 60,  # 居民满意度最低要求
            Nt,  # 游客数量非负
            tau_t,  # 旅游税非负
            0.4 - (self.k5 + self.k6 + self.k7),  # 投资系数和的约束
            Cwaste_adj,  # 确保废物处理容量为正
            Cwater_adj,  # 确保水资源容量为正
        ]
        return constraints

    def is_feasible(self, x: np.ndarray) -> bool:
        """检查解是否满足所有约束条件"""
        return all(c >= 0 for c in self.constraints(x))

    def dominates(
        self,
        obj1: Tuple[float, float, float, float, float, float],
        obj2: Tuple[float, float, float, float, float, float],
    ) -> bool:
        """检查obj1是否支配obj2"""
        return (
            all(o1 >= o2 for o1, o2 in zip(obj1[:3], obj2[:3]))
            and all(o1 <= o2 for o1, o2 in zip(obj1[3:], obj2[3:]))
            and (
                any(o1 > o2 for o1, o2 in zip(obj1[:3], obj2[:3]))
                or any(o1 < o2 for o1, o2 in zip(obj1[3:], obj2[3:]))
            )
        )

    def crossover_and_mutate(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """进行交叉和变异操作"""
        # 算术交叉
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2

        # 变异
        mutation_rate = 0.1
        if np.random.random() < mutation_rate:
            mutation_strength = 0.1
            child += np.random.normal(0, mutation_strength, size=2)

        # 确保在约束范围内
        child[0] = np.clip(child[0], 0, self.Nt_max)
        child[1] = np.clip(child[1], 0, self.tau_max)

        return child

    def pareto_optimization(
        self, population_size: int = 100, generations: int = 50
    ) -> List[np.ndarray]:
        """执行帕累托优化"""
        # 初始化种群
        population = []
        while len(population) < population_size:
            x = np.array(
                [
                    np.random.uniform(0, self.Nt_max),
                    np.random.uniform(0, self.tau_max),
                ]
            )
            if self.is_feasible(x):
                population.append(x)

        pareto_front = []
        for generation in range(generations):
            # 评估当前种群的目标函数值
            objectives = []
            for solution in population:
                if self.is_feasible(solution):
                    obj1 = self.calculate_revenue(solution)
                    obj2 = self.calculate_social_satisfaction(solution)
                    obj3 = self.calculate_environmental_quality(solution)
                    P_waste, P_water, P_e = self.calculate_investments(solution)
                    objectives.append((obj1, obj2, obj3, P_waste, P_water, P_e))
                else:
                    objectives.append(
                        (-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf)
                    )

            # 更新帕累托前沿
            for i, solution in enumerate(population):
                if not self.is_feasible(solution):
                    continue
                dominated = False
                for j, other_solution in enumerate(population):
                    if (
                        i != j
                        and self.is_feasible(other_solution)
                        and self.dominates(objectives[j], objectives[i])
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append(solution)

            # 生成新种群
            new_population = []
            while len(new_population) < population_size:
                # 锦标赛选择
                tournament = np.random.choice(len(population), 2, replace=False)
                if objectives[tournament[0]][0] > objectives[tournament[1]][0]:
                    parent1 = population[tournament[0]]
                else:
                    parent1 = population[tournament[1]]

                tournament = np.random.choice(len(population), 2, replace=False)
                if objectives[tournament[0]][1] > objectives[tournament[1]][1]:
                    parent2 = population[tournament[0]]
                else:
                    parent2 = population[tournament[1]]

                # 交叉和变异
                child = self.crossover_and_mutate(parent1, parent2)
                if self.is_feasible(child):
                    new_population.append(child)

            population = new_population

        # 移除重复解
        pareto_front = np.unique(pareto_front, axis=0)
        return pareto_front

    def simulate_system(self, x: np.ndarray, time_steps: int = 10) -> List[dict]:
        """模拟系统动态变化"""
        history = []
        current_x = x.copy()

        for t in range(time_steps):
            CO2p_adj, Cwaste_adj, Cwater_adj = self.calculate_adjusted_capacities(
                current_x
            )
            current_E = self.calculate_environmental_quality(current_x)

            state = {
                "t": t * self.dt,
                "Nt": current_x[0],
                "tau_t": current_x[1],
                "Pt": self.Pt,
                "E": current_E,
                "CO2p": CO2p_adj,
                "Cwaste": Cwaste_adj,
                "Cwater": Cwater_adj,
                "Revenue": self.calculate_revenue(current_x),
                "Satisfaction": self.calculate_social_satisfaction(current_x),
            }
            history.append(state)

        return history

    def visualize_pareto_front(self, pareto_front: List[np.ndarray]):
        """可视化帕累托前沿"""
        if len(pareto_front) == 0:
            print("No feasible solutions found!")
            return

        objectives = np.array(
            [
                [
                    self.calculate_revenue(x),
                    self.calculate_social_satisfaction(x),
                    self.calculate_environmental_quality(x),
                    *self.calculate_investments(x),
                ]
                for x in pareto_front
            ]
        )

        # 创建多个子图来显示不同的目标组合
        fig = plt.figure(figsize=(15, 10))

        # 收入-满意度-环境质量
        ax1 = fig.add_subplot(121, projection="3d")
        scatter1 = ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2])
        ax1.set_xlabel("Revenue")
        ax1.set_ylabel("Social Satisfaction")
        ax1.set_zlabel("Environmental Quality")
        ax1.set_title("Revenue-Satisfaction-Environment")

        # 三种投资的对比
        ax2 = fig.add_subplot(122, projection="3d")
        scatter2 = ax2.scatter(objectives[:, 3], objectives[:, 4], objectives[:, 5])
        ax2.set_xlabel("P_waste")
        ax2.set_ylabel("P_water")
        ax2.set_zlabel("P_e")
        ax2.set_title("Investment Distribution")

        plt.tight_layout()
        plt.show()

    def plot_simulation_results(self, history: List[dict]):
        """绘制模拟结果"""
        t = [state["t"] for state in history]
        Nt = [state["Nt"] for state in history]
        E = [state["E"] for state in history]
        CO2p = [state["CO2p"] for state in history]
        Cwaste = [state["Cwaste"] for state in history]
        Cwater = [state["Cwater"] for state in history]
        Revenue = [state["Revenue"] for state in history]
        Satisfaction = [state["Satisfaction"] for state in history]

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        # 游客数量变化
        axes[0, 0].plot(t, Nt)
        axes[0, 0].set_title("Tourist Numbers")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Nt")

        # 环境质量变化
        axes[0, 1].plot(t, E)
        axes[0, 1].set_title("Environmental Quality")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("E")

        # 容量变化
        axes[1, 0].plot(t, CO2p, label="CO2p")
        axes[1, 0].plot(t, Cwaste, label="Cwaste")
        axes[1, 0].plot(t, Cwater, label="Cwater")
        axes[1, 0].set_title("Capacity Changes")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Capacity")
        axes[1, 0].legend()

        # 收入和满意度变化
        ax2 = axes[1, 1].twinx()
        (l1,) = axes[1, 1].plot(t, Revenue, "b-", label="Revenue")
        (l2,) = ax2.plot(t, Satisfaction, "r-", label="Satisfaction")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Revenue", color="b")
        ax2.set_ylabel("Satisfaction", color="r")
        axes[1, 1].legend(handles=[l1, l2])

        plt.tight_layout()
        plt.show()

    def sensitivity_analysis(
        self, base_solution: np.ndarray, param_ranges: dict
    ) -> dict:
        """执行敏感性分析

        Args:
            base_solution: 基准解 [Nt, tau_t]
            param_ranges: 要分析的参数及其变化范围的字典
                        例如: {'k1': [0.3, 0.7], 'a1': [-2e-11, -1e-11]}

        Returns:
            包含敏感性分析结果的字典
        """
        results = {}

        # 保存原始参数值
        original_params = {}

        # 对每个参数进行分析
        for param_name, (param_min, param_max) in param_ranges.items():
            # 保存原始值
            original_value = getattr(self, param_name)
            original_params[param_name] = original_value

            # 创建参数变化序列
            param_values = np.linspace(param_min, param_max, 10)
            revenue_changes = []
            satisfaction_changes = []
            env_quality_changes = []

            # 计算基准值
            base_revenue = self.calculate_revenue(base_solution)
            base_satisfaction = self.calculate_social_satisfaction(base_solution)
            base_env_quality = self.calculate_environmental_quality(base_solution)

            # 对每个参数值计算输出变化
            for value in param_values:
                setattr(self, param_name, value)

                # 计算相对变化百分比
                revenue = self.calculate_revenue(base_solution)
                satisfaction = self.calculate_social_satisfaction(base_solution)
                env_quality = self.calculate_environmental_quality(base_solution)

                revenue_changes.append((revenue - base_revenue) / base_revenue * 100)
                satisfaction_changes.append(
                    (satisfaction - base_satisfaction) / base_satisfaction * 100
                )
                env_quality_changes.append(
                    (env_quality - base_env_quality) / base_env_quality * 100
                )

            # 恢复原始参数值
            setattr(self, param_name, original_value)

            # 存储结果
            results[param_name] = {
                "param_values": param_values,
                "revenue_changes": revenue_changes,
                "satisfaction_changes": satisfaction_changes,
                "env_quality_changes": env_quality_changes,
            }

        return results

    def plot_sensitivity_analysis(self, results: dict):
        """绘制敏感性分析结果

        Args:
            results: sensitivity_analysis方法返回的结果字典
        """
        num_params = len(results)
        fig, axes = plt.subplots(num_params, 1, figsize=(12, 5 * num_params))
        if num_params == 1:
            axes = [axes]

        for idx, (param_name, param_results) in enumerate(results.items()):
            ax = axes[idx]

            # 绘制三个指标的变化
            ax.plot(
                param_results["param_values"],
                param_results["revenue_changes"],
                label="Revenue",
                marker="o",
            )
            ax.plot(
                param_results["param_values"],
                param_results["satisfaction_changes"],
                label="Satisfaction",
                marker="s",
            )
            ax.plot(
                param_results["param_values"],
                param_results["env_quality_changes"],
                label="Environmental Quality",
                marker="^",
            )

            ax.set_xlabel(f"Parameter {param_name} value")
            ax.set_ylabel("Percentage change (%)")
            ax.set_title(f"Sensitivity Analysis for {param_name}")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create optimizer instance
    optimizer = TourismOptimization()

    # Execute Pareto optimization
    print("Performing Pareto optimization...")
    pareto_front = optimizer.pareto_optimization(population_size=100, generations=50)

    # Visualize Pareto front
    print(f"Found {len(pareto_front)} Pareto optimal solutions")
    optimizer.visualize_pareto_front(pareto_front)

    # Print all Pareto optimal solutions
    print("\nAll Pareto optimal solutions:")
    for i, solution in enumerate(pareto_front, 1):
        P_waste, P_water, P_e = optimizer.calculate_investments(solution)
        print(f"\nSolution {i}:")
        print(f"Tourist Numbers: {solution[0]:.0f}")
        print(f"Tourism Tax: {solution[1]:.2f}")
        print(f"Tourist Consumption: {optimizer.Pt:.2f}")
        print(f"Expected Revenue: {optimizer.calculate_revenue(solution):.2f}")
        print(
            f"Social Satisfaction: {optimizer.calculate_social_satisfaction(solution):.2f}"
        )
        print(
            f"Environmental Quality: {optimizer.calculate_environmental_quality(solution):.2f}"
        )
        print(f"Waste Treatment Investment: {P_waste:.2f}")
        print(f"Water Resource Investment: {P_water:.2f}")
        print(f"Environmental Investment: {P_e:.2f}")

    # 添加敏感性分析
    if len(pareto_front) > 0:
        # 选择一个基准解（例如，收入最高的解）
        base_solution = max(pareto_front, key=lambda x: optimizer.calculate_revenue(x))

        # 定义要分析的参数及其范围
        param_ranges = {
            "k1": [0.3, 0.7],  # 环保投资效率系数
            "k2": [0.1, 0.5],  # 碳排放影响系数
            "a1": [-3e-11, -1e-11],  # 游客数量对居民满意度的影响系数
            "a2": [2e-5, 7e-5],  # 旅游税对游客满意度的影响系数
        }

        # 执行敏感性分析
        print("\nPerforming sensitivity analysis...")
        sensitivity_results = optimizer.sensitivity_analysis(
            base_solution, param_ranges
        )

        # 可视化结果
        optimizer.plot_sensitivity_analysis(sensitivity_results)


if __name__ == "__main__":
    main()
