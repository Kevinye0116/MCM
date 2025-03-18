import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize


class TourismOptimizer:
    def __init__(self):
        # 完全保留原始参数和方程
        self.Nmax = 2_000_000  # 最大游客数（硬编码）
        self.CO2max = 250_000  # 最大CO2排放量（吨）
        self.CO2p = 0.184  # 每位游客的CO2排放量（吨/人）
        self.Cbase = 500  # 基础环境容量
        self.Cwater = 18_900_000  # 水资源容量
        self.Cwaste = 2_549_110  # 废弃物容量
        self.Pt = 190  # 游客平均消费（美元/人）
        self.Restandrand = 500_000_000  # 收入标准化常数
        self.Sbase = 100  # 社会满意度基准值

        # 原始权重系数
        self.k1 = 0.5  # 环境权重
        self.k2 = 0.3  # 废弃物容量权重
        self.k3 = 0.2  # 水资源容量权重

        # 原始社会满意度函数参数
        self.a1 = -1.975284171471327e-11
        self.a2 = 4.772497706879391e-05
        self.b1 = 39.266042633247565

        # 基础设施动态增长系数（完全保留原始定义）
        self.alpha1 = 0.02  # 废弃物容量增长率
        self.alpha2 = 0.05  # 水资源容量增长率
        self.alpha3 = 0.05  # 基础环保容量增长率

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """完全保留原始投资计算逻辑"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re

        # 基础设施容量更新（完全保留原始公式）
        self.Cwaste += self.alpha1 * P_waste / 1e6
        self.Cwater += self.alpha2 * P_water / 1e6
        self.Cbase += self.alpha3 * P_e / 1e6
        return P_waste, P_water, P_e

    def objective(self, x):
        """完全保留原始目标函数"""
        Nt, tau_t, k5, k6, k7 = x
        Re = (self.Pt * Nt) / self.Restandrand
        E_env = (
            self.k1 * (self.CO2p * Nt - self.Cbase) / self.CO2max
            + self.k2 * Nt / self.Cwaste
            + self.k3 * Nt / self.Cwater
        )
        S_res = (self.a1 * Nt**2 + self.a2 * Nt + self.b1) / self.Sbase
        return -(Re + S_res - E_env)  # 负值用于最小化

    def constraints(self):
        """完全保留原始约束条件"""

        def constraint_func(x):
            Nt, tau_t, k5, k6, k7 = x
            return [
                self.Pt * Nt,  # Re >= 0
                0.08 - tau_t,  # tau_t <= 8%
                self.Nmax - Nt,  # Nt <= 2,000,000
                Nt - 100_000,  # Nt >= 100,000
                self.CO2max - Nt * self.CO2p,  # CO2排放约束
                0.4 - (k5 + k6 + k7),  # 总投资比例 <= 40%
                (k5 + k6 + k7) - 0.15,  # 总投资比例 >= 15%
                k5 - 0.05,
                k6 - 0.05,
                k7 - 0.05,  # 各投资比例 >=5%
            ]

        return NonlinearConstraint(constraint_func, 0, np.inf)

    def optimize(self):
        """混合优化策略：差分进化（全局） + SLSQP（局部）"""
        # 变量边界（完全保留原始范围）
        bounds = [
            (100_000, self.Nmax),  # Nt ∈ [100k, 2M]
            (0.02, 0.08),  # tau_t ∈ [2%, 8%]
            (0.05, 0.2),
            (0.05, 0.2),
            (0.05, 0.2),  # k5, k6, k7 ∈ [5%, 20%]
        ]

        # 第一阶段：全局优化（差分进化）
        print("Global optimization with Differential Evolution...")
        result_de = differential_evolution(
            self.objective,
            bounds,
            strategy="best1bin",
            maxiter=200,  # 增加全局搜索迭代次数
            popsize=30,  # 增大种群规模
            mutation=(0.7, 1.5),  # 扩大变异范围
            recombination=0.9,  # 提高交叉概率
            constraints=self.constraints(),
            polish=False,  # 禁用自动局部优化
        )

        # 第二阶段：局部优化（SLSQP）
        print("\nLocal refinement with SLSQP...")
        result_local = minimize(
            self.objective,
            result_de.x,  # 以全局解为初始值
            method="SLSQP",
            bounds=bounds,
            constraints=self.constraints(),
            options={"maxiter": 5000, "ftol": 1e-15},  # 提高局部优化精度
        )

        return result_local if result_local.success else result_de

    def analyze_solution(self, result):
        """完全保留原始结果分析逻辑"""
        Nt, tau_t, k5, k6, k7 = result.x
        print("\nOptimization Results:")
        print(f"- Tourists (Nt): {Nt:.0f}")
        print(f"- Tax rate (tau_t): {tau_t*100:.2f}%")
        print(f"- Investment ratios: Waste {k5:.2f}, Water {k6:.2f}, Env {k7:.2f}")
        print(f"- Objective value (Z): {-result.fun:.2f}")

        # 约束合规性检查
        print("\nConstraint Compliance:")
        print(
            f"CO2 emissions: {Nt*self.CO2p:.0f}/{self.CO2max} → {'OK' if Nt*self.CO2p <= self.CO2max else 'Violated'}"
        )
        print(
            f"Total investment ratio: {k5+k6+k7:.2f} (0.15≤sum≤0.4) → {'OK' if 0.15<=k5+k6+k7<=0.4 else 'Violated'}"
        )


if __name__ == "__main__":
    optimizer = TourismOptimizer()
    result = optimizer.optimize()
    optimizer.analyze_solution(result)
