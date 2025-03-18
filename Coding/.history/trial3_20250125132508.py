from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class TourismOptimization:
    def __init__(self):
        # 模型系数
        self.k1 = 0.1  # 环保投资效率系数
        self.k2 = 0.05  # 碳排放影响系数
        self.k3 = 0.2  # 自然恢复系数
        self.k4 = 0.15  # 基础设施发展系数
        self.k5 = 0.3  # 基础设施投资系数
        self.k6 = 0.25  # 环保投资系数

        # 社会满意度参数
        self.a1 = 0.01  # 游客数量对居民满意度的影响系数
        self.a2 = 0.02  # 旅游税对游客满意度的影响系数
        self.a3 = 0.015  # 消费水平对游客满意度的影响系数
        self.b1 = 100  # 居民基础满意度
        self.b2 = 100  # 游客基础满意度

        # 发展参数
        self.gamma = 0.1  # 游客增长率
        self.delta = 0.05  # 环境影响因子

        # 约束条件上限
        self.Nt_max = 5000000  # 最大游客数量
        self.tau_max = 100  # 最大旅游税
        self.Pt_max = 10000  # 最大游客消费
        self.CO2_max = 500000  # 最大碳排放量
        self.E_min = 50  # 最小环境质量指数

        # 其他参数
        self.Ch = 1000  # 隐性成本
        self.CO2p = 0.5  # 人均碳排放量
        self.Rnature = 100  # 自然恢复能力基准值

        # 系统动态参数
        self.dt = 0.1  # 时间步长
        self.E = self.E_min  # 初始环境质量
        self.Cinfra = 1000  # 初始基础设施容量

    def calculate_revenue(self, x: np.ndarray) -> float:
        """计算总收入"""
        Nt, tau_t, Pt = x
        Pe = self.k6 * tau_t * Nt  # 环保投资
        Pb = self.k5 * tau_t * Nt  # 基础设施投资
        return (tau_t + Pt) * Nt - Pe - Pb - self.Ch

    def calculate_social_satisfaction(self, x: np.ndarray) -> float:
        """计算社会满意度"""
        Nt, tau_t, Pt = x
        S_residents = -self.a1 * Nt + self.b1  # 居民满意度
        S_tourists = -self.a2 * tau_t - self.a3 * Pt + self.b2  # 游客满意度
        return S_residents + S_tourists

    def calculate_environmental_quality(self, x: np.ndarray) -> float:
        """计算环境质量指数"""
        Nt, tau_t, Pt = x
        Pe = self.k6 * tau_t * Nt  # 环保投资
        return self.k1 * Pe - self.k2 * (self.CO2p * Nt) + self.k3 * self.Rnature

    def calculate_dE_dt(self, x: np.ndarray) -> float:
        """计算环境质量的变化率 dE/dt"""
        Nt, tau_t, Pt = x
        Pe = self.k6 * tau_t * Nt  # 环保投资
        return self.k1 * Pe - self.k2 * (self.CO2p * Nt) + self.k3 * self.Rnature

    def calculate_dCinfra_dt(self, x: np.ndarray) -> float:
        """计算基础设施容量的变化率 dCinfra/dt"""
        Nt, tau_t, Pt = x
        Pb = self.k5 * tau_t * Nt  # 基础设施投资
        return self.k4 * Pb

    def calculate_next_Nt(self, x: np.ndarray, E: float) -> float:
        """计算下一时刻的游客数量"""
        Nt, tau_t, Pt = x
        Stourists = -self.a2 * tau_t - self.a3 * Pt + self.b2
        return Nt * (1 + self.gamma * Stourists / 100) - self.delta * (self.E_min - E)

    def constraints(self, x: np.ndarray) -> List[float]:
        """检查所有约束条件"""
        Nt, tau_t, Pt = x
        return [
            self.Nt_max - Nt,  # 游客数量上限约束
            self.tau_max - tau_t,  # 旅游税上限约束
            self.Pt_max - Pt,  # 消费上限约束
            self.CO2_max - Nt * self.CO2p,  # 碳排放约束
            -self.a1 * Nt + self.b1 - 60,  # 居民满意度最低要求
            Nt,  # 游客数量非负
            tau_t,  # 旅游税非负
            Pt,  # 消费非负
        ]

    def is_feasible(self, x: np.ndarray) -> bool:
        """检查解是否满足所有约束条件"""
        return all(c >= 0 for c in self.constraints(x))

    def dominates(
        self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]
    ) -> bool:
        """检查obj1是否支配obj2"""
        return all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and any(
            o1 > o2 for o1, o2 in zip(obj1, obj2)
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
            child += np.random.normal(0, mutation_strength, size=3)

        # 确保在约束范围内
        child[0] = np.clip(child[0], 0, self.Nt_max)
        child[1] = np.clip(child[1], 0, self.tau_max)
        child[2] = np.clip(child[2], 0, self.Pt_max)

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
                    np.random.uniform(0, self.Pt_max),
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
                    objectives.append((obj1, obj2, obj3))
                else:
                    objectives.append((-np.inf, -np.inf, -np.inf))

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
        current_E = self.E
        current_Cinfra = self.Cinfra

        for t in range(time_steps):
            # 记录当前状态
            state = {
                "t": t * self.dt,
                "Nt": current_x[0],
                "tau_t": current_x[1],
                "Pt": current_x[2],
                "E": current_E,
                "Cinfra": current_Cinfra,
                "Revenue": self.calculate_revenue(current_x),
                "Satisfaction": self.calculate_social_satisfaction(current_x),
            }
            history.append(state)

            # 计算变化率
            dE = self.calculate_dE_dt(current_x) * self.dt
            dCinfra = self.calculate_dCinfra_dt(current_x) * self.dt

            # 更新状态变量
            current_E += dE
            current_Cinfra += dCinfra
            next_Nt = self.calculate_next_Nt(current_x, current_E)

            # 更新游客数量，保持其他控制变量不变
            current_x[0] = np.clip(next_Nt, 0, self.Nt_max)

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
                ]
                for x in pareto_front
            ]
        )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            objectives[:, 0], objectives[:, 1], objectives[:, 2], c="b", marker="o"
        )

        ax.set_xlabel("Revenue")
        ax.set_ylabel("Social Satisfaction")
        ax.set_zlabel("Environmental Quality")
        ax.set_title("Pareto Front")

        plt.show()

    def plot_simulation_results(self, history: List[dict]):
        """绘制模拟结果"""
        t = [state["t"] for state in history]
        Nt = [state["Nt"] for state in history]
        E = [state["E"] for state in history]
        Cinfra = [state["Cinfra"] for state in history]
        Revenue = [state["Revenue"] for state in history]
        Satisfaction = [state["Satisfaction"] for state in history]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

        # 基础设施容量变化
        axes[1, 0].plot(t, Cinfra)
        axes[1, 0].set_title("Infrastructure Capacity")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Cinfra")

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


def main():
    # 设置随机种子以保证结果可重复
    np.random.seed(42)

    # 创建优化器实例
    optimizer = TourismOptimization()

    # 执行帕累托优化
    print("正在进行帕累托优化...")
    pareto_front = optimizer.pareto_optimization(population_size=100, generations=50)

    # 可视化帕累托前沿
    print(f"找到 {len(pareto_front)} 个帕累托最优解")
    optimizer.visualize_pareto_front(pareto_front)

    # 选择一个示例解进行系统动态模拟
    if len(pareto_front) > 0:
        # 选择收入最高的解作为示例
        example_solution = max(
            pareto_front, key=lambda x: optimizer.calculate_revenue(x)
        )

        print("\n选定的示例解：")
        print(f"游客数量: {example_solution[0]:.0f}")
        print(f"旅游税: {example_solution[1]:.2f}")
        print(f"游客消费: {example_solution[2]:.2f}")
        print(f"预期收入: {optimizer.calculate_revenue(example_solution):.2f}")
        print(
            f"社会满意度: {optimizer.calculate_social_satisfaction(example_solution):.2f}"
        )
        print(
            f"环境质量: {optimizer.calculate_environmental_quality(example_solution):.2f}"
        )

        # 模拟系统动态变化
        print("\n正在模拟系统动态变化...")
        history = optimizer.simulate_system(example_solution, time_steps=50)

        # 可视化模拟结果
        optimizer.plot_simulation_results(history)
    else:
        print("未找到可行解！")


if __name__ == "__main__":
    main()
