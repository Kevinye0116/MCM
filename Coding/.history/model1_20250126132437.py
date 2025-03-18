import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint


class TourismOptimizer:
    def __init__(self):
        # 原始初始数据（完全恢复）
        self.Nmax = 2_000_000  # 最大游客数（硬编码）
        self.CO2max = 250_000  # 最大CO2排放量（吨）
        self.CO2p = 0.184  # 每位游客的CO2排放量（吨/人）
        self.Cbase = 500  # 基础环境容量
        self.Cwater = 18_900_000  # 水资源容量
        self.Cwaste = 2_549_110  # 废弃物容量
        self.Pt = 190  # 游客平均消费（美元/人）
        self.Restandrand = 500_000_000  # 收入标准化常数
        self.Sbase = 100  # 社会满意度基准值

        # 原始权重系数（恢复为初始值）
        self.k1 = 0.5  # 环境权重
        self.k2 = 0.3  # 废弃物容量权重
        self.k3 = 0.2  # 水资源容量权重

        # 原始社会满意度函数参数（恢复为初始值）
        self.a1 = -1.975284171471327e-11
        self.a2 = 4.772497706879391e-05
        self.b1 = 39.266042633247565

        # 基础设施动态增长系数（保持原始定义）
        self.alpha1 = 0.02  # 废弃物容量增长率
        self.alpha2 = 0.05  # 水资源容量增长率
        self.alpha3 = 0.05  # 基础环保容量增长率

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """计算投资并更新基础设施容量（保持原始逻辑）"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re

        # 基础设施容量更新（按原始公式）
        self.Cwaste += self.alpha1 * P_waste / 1e6  # 假设每百万投资提升容量
        self.Cwater += self.alpha2 * P_water / 1e6
        self.Cbase += self.alpha3 * P_e / 1e6
        return P_waste, P_water, P_e

    def objective(self, x):
        """目标函数：最大化 Z = Re + S - E（完全恢复原始公式）"""
        Nt, tau_t, k5, k6, k7 = x

        # 计算收入（原始公式）
        Re = (self.Pt * Nt) / self.Restandrand

        # 计算环境成本（原始公式）
        E_env = (
            self.k1 * (self.CO2p * Nt - self.Cbase) / self.CO2max
            + self.k2 * Nt / self.Cwaste
            + self.k3 * Nt / self.Cwater
        )

        # 计算社会满意度（原始公式）
        S_res = (self.a1 * Nt**2 + self.a2 * Nt + self.b1) / self.Sbase

        # 返回负值（因优化器为最小化）
        return -(Re + S_res - E_env)

    def constraints(self):
        """定义非线性约束（完全恢复原始约束）"""

        def constraint_func(x):
            Nt, tau_t, k5, k6, k7 = x
            # 所有约束条件值（需>=0）
            return [
                self.Pt * Nt,  # Re >= 0
                0.08 - tau_t,  # tau_t <= 8%
                self.Nmax - Nt,  # Nt <= 2,000,000（硬编码）
                Nt - 100_000,  # Nt >= 100,000
                self.CO2max - Nt * self.CO2p,  # CO2排放约束
                0.4 - (k5 + k6 + k7),  # 总投资比例 <= 40%
                (k5 + k6 + k7) - 0.15,  # 总投资比例 >= 15%
                k5 - 0.05,
                k6 - 0.05,
                k7 - 0.05,  # 各投资比例 >=5%
            ]

        # 所有约束需>=0
        return NonlinearConstraint(constraint_func, 0, np.inf)

    def optimize(self):
        """混合优化：差分进化（全局） + SLSQP（局部）"""
        # 变量边界（完全恢复原始范围）
        bounds = [
            (100_000, self.Nmax),  # Nt ∈ [100,000, 2,000,000]
            (0.02, 0.08),  # tau_t ∈ [2%, 8%]
            (0.05, 0.2),
            (0.05, 0.2),
            (0.05, 0.2),  # k5, k6, k7 ∈ [5%, 20%]
        ]

        # 全局优化：差分进化
        print("Running global optimization (differential evolution)...")
        result_de = differential_evolution(
            self.objective,
            bounds,
            strategy="best1bin",
            maxiter=50,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            constraints=self.constraints(),
            polish=False,  # 禁用局部优化（后续手动执行）
        )

        # 局部优化：SLSQP（以全局解为初始值）
        print("\nRefining with local optimization (SLSQP)...")
        result_local = minimize(
            self.objective,
            result_de.x,
            method="SLSQP",
            bounds=bounds,
            constraints=self.constraints(),
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        # 返回最优解
        return result_local if result_local.success else result_de

    def analyze_solution(self, result):
        """结果分析与可行性检查（保持原始逻辑）"""
        Nt, tau_t, k5, k6, k7 = result.x
        print("\n优化结果:")
        print(f"- 游客数量 (Nt): {Nt:.0f} 人")
        print(f"- 税率 (tau_t): {tau_t*100:.2f}%")
        print(f"- 投资比例: 废弃物 {k5:.2f}, 水资源 {k6:.2f}, 环保 {k7:.2f}")
        print(f"- 总目标值 Z: {-result.fun:.2f}")

        # 检查关键约束
        print("\n约束合规性检查:")
        print(
            f"CO2排放: {Nt * self.CO2p:.1f} 吨 (上限 {self.CO2max}) → {'合规' if Nt * self.CO2p <= self.CO2max else '违规'}"
        )
        print(
            f"游客数量上限: {Nt:.0f} (上限 {self.Nmax}) → {'合规' if Nt <= self.Nmax else '违规'}"
        )
        print(
            f"投资比例总和: {k5 + k6 + k7:.2f} (应在[0.15, 0.4]) → {'合规' if 0.15 <= (k5 + k6 + k7) <= 0.4 else '违规'}"
        )


# 运行优化
if __name__ == "__main__":
    optimizer = TourismOptimizer()
    result = optimizer.optimize()
    optimizer.analyze_solution(result)
