from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution


class TourismOptimization:
    def __init__(self):
        # 模型系数
        self.k1 = 0.4  # 环保投资效率系数
        self.k2 = 0.3  # 碳排放影响系数
        self.k3 = 0.3  # 自然恢复系数
        self.k4 = 0.15  # 基础设施发展系数
        self.k5 = 0.3  # 基础设施投资系数
        self.k6 = 0.25  # 环保投资系数

        # 社会满意度参数
        self.a1 = -1.975284171471327e-11  # 游客数量对居民满意度的影响系数
        self.a2 = 4.772497706879391e-05  # 旅游税对游客满意度的影响系数
        self.a3 = 0.015  # 消费水平对游客满意度的影响系数
        self.b1 = 39.266042633247565  # 居民基础满意度
        self.b2 = 60  # 游客基础满意度

        # 约束条件上限
        self.Nt_max = 5000000  # 最大游客数量
        self.tau_max = 10  # 最大旅游税
        self.Pt_max = 5000  # 最大游客消费
        self.CO2_max = 8000000  # 最大碳排放量
        self.E = 0  # 最小环境质量指数

        # 其他参数
        self.CO2p = 9.38  # 人均碳排放量
        self.dt = 0.1  # 时间步长

        # 新增参数
        self.Cwater = 18900000  # 水资源容量
        self.Cwaste = 2549110  # 废物处理容量
        self.a4 = 0.01  # 游客数量对游客满意度的影响系数
        self.b3 = 10  # 废物处理基准增长率

        # 目标函数权重
        self.w1 = 0.4  # 收入权重
        self.w2 = 0.3  # 社会满意度权重
        self.w3 = 0.3  # 环境质量权重

    def calculate_revenue(self, x: np.ndarray) -> float:
        """计算总收入"""
        Nt, tau_t, _ = x
        return 200 * Nt

    def calculate_social_satisfaction(self, x: np.ndarray) -> float:
        """计算社会满意度"""
        Nt, tau_t, _ = x
        S_residents = self.a1 * Nt * Nt + self.a2 * Nt + self.b1
        return np.clip(S_residents, 0, 100)

    def calculate_environmental_quality(self, x: np.ndarray) -> float:
        """计算环境质量指数"""
        Nt, tau_t, _ = x
        return (
            self.k1 * self.CO2p * Nt / self.CO2_max
            + self.k2 * Nt / self.Cwaste
            + self.k3 * Nt / self.Cwater
        )

    def objective_function(self, x: np.ndarray) -> float:
        """单目标函数：最大化加权和"""
        if not self.is_feasible(x):
            return float("inf")  # 不可行解返回极大值

        # 归一化各个目标
        revenue = self.calculate_revenue(x) / (self.Pt_max * self.Nt_max)
        satisfaction = self.calculate_social_satisfaction(x) / 100
        env_quality = self.calculate_environmental_quality(x)

        # 计算加权和（由于环境质量是越小越好，所以用1-env_quality）
        return -(
            self.w1 * revenue + self.w2 * satisfaction + self.w3 * (1 - env_quality)
        )

    def constraints(self, x: np.ndarray) -> List[float]:
        """检查所有约束条件"""
        Nt, tau_t, _ = x
        return [
            self.Nt_max - Nt,  # 游客数量上限约束
            self.tau_max - tau_t,  # 旅游税上限约束
            self.Pt_max - _,  # 消费上限约束
            self.CO2_max - Nt * self.CO2p,  # 碳排放约束
            self.a1 * Nt * Nt + self.a2 * Nt + self.b1 - 60,  # 居民满意度最低要求
            Nt,  # 游客数量非负
            tau_t,  # 旅游税非负
            _,  # 消费非负
        ]

    def is_feasible(self, x: np.ndarray) -> bool:
        """检查解是否满足所有约束条件"""
        return all(c >= 0 for c in self.constraints(x))

    def optimize(self) -> tuple:
        """使用差分进化算法进行单目标优化"""
        bounds = [(0, self.Nt_max), (0, self.tau_max), (200, 200)]

        result = differential_evolution(
            self.objective_function,
            bounds,
            strategy="best1bin",
            maxiter=100,
            popsize=20,
            seed=42,
        )

        return (
            result.x,
            -result.fun,
        )  # 返回最优解和目标函数值（取反因为我们在最小化负的目标函数）

    def simulate_system(self, x: np.ndarray, time_steps: int = 10) -> List[dict]:
        """模拟系统动态变化"""
        history = []
        current_x = x.copy()
        current_Cwaste = self.Cwaste

        for t in range(time_steps):
            current_E = self.calculate_environmental_quality(current_x)

            state = {
                "t": t * self.dt,
                "Nt": current_x[0],
                "tau_t": current_x[1],
                "Pt": current_x[2],
                "E": current_E,
                "Cwaste": current_Cwaste,
                "Revenue": self.calculate_revenue(current_x),
                "Satisfaction": self.calculate_social_satisfaction(current_x),
            }
            history.append(state)

            # 更新状态
            dCwaste = (
                self.k4 * self.k5 * current_x[1] * current_x[0] + self.b3
            ) * self.dt
            current_Cwaste += dCwaste

        return history

    def plot_simulation_results(self, history: List[dict]):
        """绘制模拟结果"""
        t = [state["t"] for state in history]
        Nt = [state["Nt"] for state in history]
        E = [state["E"] for state in history]
        Cwaste = [state["Cwaste"] for state in history]
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

        # 废物处理容量变化
        axes[2, 0].plot(t, Cwaste)
        axes[2, 0].set_title("Waste Treatment Capacity")
        axes[2, 0].set_xlabel("Time")
        axes[2, 0].set_ylabel("Cwaste")

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
    # 设置随机种子
    np.random.seed(42)

    # 创建优化器实例
    optimizer = TourismOptimization()

    # 执行单目标优化
    print("Performing single-objective optimization...")
    solution, objective_value = optimizer.optimize()

    # 打印优化结果
    print("\nOptimal solution found:")
    print(f"Tourist Numbers: {solution[0]:.0f}")
    print(f"Tourism Tax: {solution[1]:.2f}")
    print(f"Tourist Consumption: {solution[2]:.2f}")
    print(f"Objective Value: {objective_value:.2f}")
    print(f"Expected Revenue: {optimizer.calculate_revenue(solution):.2f}")
    print(
        f"Social Satisfaction: {optimizer.calculate_social_satisfaction(solution):.2f}"
    )
    print(
        f"Environmental Quality: {optimizer.calculate_environmental_quality(solution):.2f}"
    )

    # 模拟系统动态
    print("\nSimulating system dynamics...")
    history = optimizer.simulate_system(solution, time_steps=50)

    # 可视化模拟结果
    optimizer.plot_simulation_results(history)


if __name__ == "__main__":
    main()
