import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sympy import print_glsl


class TourismOptimizer:
    def __init__(self):
        # Constants
        self.Nmax = 2000000  # Maximum tourists per year
        self.CO2max = 250000  # Maximum CO2 emissions
        self.Cbase = 500
        self.Cwater = 18900000  # Waste capacity
        self.Cwaste = 2549110  # Water capacity
        self.Pt = 190  # Average spending per tourist
        self.CO2p = 0.184  # Carbon emissions per person
        self.Ptmax = 200  # Maximum infrastructure investment
        self.Restandrand = 500000000
        self.Sbase = 100

        # Coefficients
        self.k1 = 0.5  # Environmental impact weights
        self.k2 = 0.3
        self.k3 = 0.2
        self.a1 = -1.975284171471327e-11  # Resident satisfaction quadratic term
        self.a2 = 4.772497706879391e-05  # Resident satisfaction linear term
        self.b1 = 39.266042633247565  # Resident satisfaction constant
        self.alpha1 = 0.02  # Waste capacity growth rate
        self.alpha2 = 0.05  # Water capacity growth rate
        self.alpha3 = 0.05  # Environmental protection effectiveness

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations based on revenue"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def objective(self, x):
        """
        Objective function to maximize Z = Re + S - E
        x = [Nt, tau_t, k5, k6, k7]
        """
        Nt, tau_t, k5, k6, k7 = x

        # Calculate revenue (equation 1)
        Re = self.Pt * Nt / self.Restandrand

        P = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        Cbase = self.Cbase + self.alpha1 * P[2]
        Cwaste = self.Cwaste + self.alpha2 * P[0]
        Cwater = self.Cwater + self.alpha2 * P[1]

        # Calculate environmental impact (equation 2)
        E = (
            self.k1 * (self.CO2p * Nt - Cbase) / self.CO2max
            + self.k2 * Nt / Cwaste
            + self.k3 * Nt / Cwater
        )

        # Calculate societal satisfaction (equation 3)
        Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = Sresidents / self.Sbase

        P = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        # Return negative of objective (equation 6) for minimization
        return -(Re + S - E)

    def constraints(self):
        """Define optimization constraints"""
        cons = [
            # Financial constraints (equation 7)
            {"type": "ineq", "fun": lambda x: self.Pt * x[0]},  # Re >= 0
            {"type": "ineq", "fun": lambda x: 0.08 - x[1]},  # tau_t <= 8%
            # Tourism constraints (equation 8)
            {"type": "ineq", "fun": lambda x: self.Nmax - x[0]},  # Nt <= Nmax
            {"type": "ineq", "fun": lambda x: x[0] - 100000},  # Nt >= 100000
            # Environmental constraints (equation 9)
            {"type": "ineq", "fun": lambda x: self.CO2max - x[0] * self.CO2p},
            # Investment ratio constraints (equation 5)
            {
                "type": "ineq",
                "fun": lambda x: 0.4 - (x[2] + x[3] + x[4]),
            },  # 总投资比例 <= 0.4
            {
                "type": "ineq",
                "fun": lambda x: x[2] + x[3] + x[4] - 0.15,
            },  # 总投资比例 >= 0.15
            # 各投资比例的最小值约束
            {"type": "ineq", "fun": lambda x: x[2] - 0.05},  # k5 >= 0.05
            {"type": "ineq", "fun": lambda x: x[3] - 0.05},  # k6 >= 0.05
            {"type": "ineq", "fun": lambda x: x[4] - 0.05},  # k7 >= 0.05
            # 基建承载量约束
            # {"type": "ineq", "fun": lambda x:  self.Cwaste * 1.2 / 365 - 0.012 * x[0]},
            # {"type": "ineq", "fun": lambda x: self.Cwater * 1.2 / 365  - 0.012 * x[0]},
        ]
        return cons

    def optimize(self):
        """优化函数"""
        # 更保守的初始值
        x0 = [5000000, 0.04, 0.1, 0.1, 0.1]  # [Nt, tau_t, k5, k6, k7]

        bounds = [
            (100000, self.Nmax),  # Nt: 设置更合理的下限
            (0.02, 0.08),  # tau_t: 确保税率有合理的下限
            (0.05, 0.2),  # k5: 收窄投资比例范围
            (0.05, 0.2),  # k6
            (0.05, 0.2),  # k7
        ]

        # 添加更多优化选项以提高稳定性
        result = minimize(
            self.objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=self.constraints(),
            options={
                "maxiter": 2000,  # 增加最大迭代次数
                "ftol": 1e-8,  # 提高精度
                "disp": True,  # 显示优化过程
                "eps": 1e-10,  # 减小梯度计算步长
                "finite_diff_rel_step": 1e-8,  # 添加有限差分相对步长
            },
        )

        # 如果优化失败，尝试不同的初始值
        if not result.success:
            print(
                "\nFirst optimization attempt failed, trying with different initial values..."
            )
            print(result.message)
            x0_alt = [300000, 0.03, 0.15, 0.15, 0.1]
            result = minimize(
                self.objective,
                x0_alt,
                method="SLSQP",
                bounds=bounds,
                constraints=self.constraints(),
                options={"maxiter": 2000, "ftol": 1e-8, "disp": True, "eps": 1e-10},
            )

        return result

    def analyze_solution(self, result):
        """Analyze and print optimization results"""
        Nt, tau_t, k5, k6, k7 = result.x

        print("\nOptimization Results:")
        print(f"Number of tourists (Nt): {Nt:.0f}")
        print(f"Tourist tax rate (tau_t): {tau_t*100:.2f}%")
        print(f"Waste management investment ratio (k5): {k5:.3f}")
        print(f"Water management investment ratio (k6): {k6:.3f}")
        print(f"Environmental protection investment ratio (k7): {k7:.3f}")
        print(f"Total investment ratio: {(k5+k6+k7):.3f}")
        print("\nFinal Capacities:")
        print(f"Waste capacity: {self.Cwaste:.0f}")
        print(f"Water capacity: {self.Cwater:.0f}")
        print(f"Base environmental capacity: {self.Cbase:.0f}")
        print(f"\nObjective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")

    def check_feasibility(self, x):
        """检查解的可行性"""
        print("\nChecking solution feasibility:")
        Nt, tau_t, k5, k6, k7 = x

        # 检查基本约束
        print(
            f"Tourist number: {Nt} ({'OK' if 100000 <= Nt <= self.Nmax else 'Violated'})"
        )
        print(f"Tax rate: {tau_t*100}% ({'OK' if 0 <= tau_t <= 0.08 else 'Violated'})")

        # 检查投资比例
        total_inv = k5 + k6 + k7
        print(
            f"Total investment ratio: {total_inv} ({'OK' if 0.15 <= total_inv <= 0.4 else 'Violated'})"
        )
        print(f"Individual ratios: k5={k5}, k6={k6}, k7={k7}")

        # 检查环境约束
        co2_emissions = Nt * self.CO2p
        print(
            f"CO2 emissions: {co2_emissions} ({'OK' if co2_emissions <= self.CO2max else 'Violated'})"
        )


# Run optimization
if __name__ == "__main__":
    optimizer = TourismOptimizer()
    result = optimizer.optimize()
    optimizer.analyze_solution(result)
