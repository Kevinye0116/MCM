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
        self.Nt_max = 10000  # 最大游客数量
        self.tau_max = 100  # 最大旅游税
        self.Pt_max = 1000  # 最大游客消费
        self.CO2_max = 500000  # 最大碳排放量
        self.E_min = 50  # 最小环境质量指数

        # 其他参数
        self.Ch = 1000  # 隐性成本
        self.CO2p = 0.5  # 人均碳排放量
        self.Rnature = 100  # 自然恢复能力基准值

    def calculate_revenue(self, x: np.ndarray) -> float:
        """
        计算总收入
        Args:
            x: [Nt, tau_t, Pt] - 游客数量、旅游税、人均消费
        Returns:
            总收入
        """
        Nt, tau_t, Pt = x
        Pe = self.k6 * tau_t * Nt  # 环保投资
        Pb = self.k5 * tau_t * Nt  # 基础设施投资
        return (tau_t + Pt) * Nt - Pe - Pb - self.Ch

    def calculate_social_satisfaction(self, x: np.ndarray) -> float:
        """
        计算社会满意度
        Args:
            x: [Nt, tau_t, Pt] - 游客数量、旅游税、人均消费
        Returns:
            总体社会满意度
        """
        Nt, tau_t, Pt = x
        S_residents = -self.a1 * Nt + self.b1  # 居民满意度
        S_tourists = -self.a2 * tau_t - self.a3 * Pt + self.b2  # 游客满意度
        return S_residents + S_tourists

    def calculate_environmental_quality(self, x: np.ndarray) -> float:
        """
        计算环境质量指数
        Args:
            x: [Nt, tau_t, Pt] - 游客数量、旅游税、人均消费
        Returns:
            环境质量指数
        """
        Nt, tau_t, Pt = x
        Pe = self.k6 * tau_t * Nt  # 环保投资
        return self.k1 * Pe - self.k2 * (self.CO2p * Nt) + self.k3 * self.Rnature

    def constraints(self, x: np.ndarray) -> List[float]:
        """
        检查所有约束条件
        Args:
            x: [Nt, tau_t, Pt] - 游客数量、旅游税、人均消费
        Returns:
            约束条件列表，所有值应≥0表示满足约束
        """
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
        """
        检查解是否满足所有约束条件
        """
        return all(c >= 0 for c in self.constraints(x))

    def dominates(
        self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]
    ) -> bool:
        """
        检查obj1是否支配obj2
        Args:
            obj1, obj2: (收入, 满意度, 环境质量)
        Returns:
            True如果obj1支配obj2
        """
        return all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and any(
            o1 > o2 for o1, o2 in zip(obj1, obj2)
        )

    def crossover_and_mutate(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """
        进行交叉和变异操作
        Args:
            parent1, parent2: 父代解
        Returns:
            新的子代解
        """
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
        """
        执行帕累托优化
        Args:
            population_size: 种群大小
            generations: 迭代代数
        Returns:
            帕累托最优解集
        """
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

    def visualize_pareto_front(self, pareto_front: List[np.ndarray]):
        """
        可视化帕累托前沿
        Args:
            pareto_front: 帕累托最优解集
        """
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


def main():
    # 设置随机种子以保证结果可重复
    np.random.seed(42)

    # 创建优化器实例
    optimizer = TourismOptimization()

    # 运行优化
    print("Running optimization...")
    pareto_front = optimizer.pareto_optimization(population_size=100, generations=50)

    # 输出结果
    print(f"\nFound {len(pareto_front)} best solutions")
    print("\nShow top 20 solutions:")
    for i, solution in enumerate(pareto_front[:20]):
        Nt, tau_t, Pt = solution
        revenue = optimizer.calculate_revenue(solution)
        satisfaction = optimizer.calculate_social_satisfaction(solution)
        environment = optimizer.calculate_environmental_quality(solution)

        print(f"\nSolution {i+1}:")
        print(f"Tourists: {Nt:.0f}")
        print(f"Tax: {tau_t:.2f}")
        print(f"Expenses per tourist: {Pt:.2f}")
        print(f"Total revenue: {revenue:.2f}")
        print(f"Social satisfaction: {satisfaction:.2f}")
        print(f"Environmental quality: {environment:.2f}")

    # 可视化结果
    optimizer.visualize_pareto_front(pareto_front)


if __name__ == "__main__":
    main()
