from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.optimize import minimize


class ImprovedTourismOptimizer:
    def __init__(self):
        # Keep all original constants and coefficients
        self.Nmax = 3000000
        self.CO2max = 320000
        self.Cbase = 50000
        self.Cwater = 18900000
        self.Cwaste = 2549110
        self.Pt = 190
        self.CO2p = 0.184
        self.Ptmax = 200
        self.Restandrand = 400000000
        self.Sbase = 100
        self.k1 = 0.4
        self.k2 = 0.3
        self.k3 = 0.3
        self.a1 = -1.97e-11
        self.a2 = 4.77e-05
        self.b1 = 39.27
        self.alpha1 = 0.0001
        self.alpha2 = 0.0005
        self.alpha3 = 0.0005

        # Add penalty coefficients for constraint violation
        self.penalty_coefficient = 1e6

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations based on revenue"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def constraint_violation_penalty(self, x):
        """Calculate penalty for constraint violations"""
        Nt, tau_t, k5, k6, k7 = x
        penalty = 0

        # Tourism constraints
        if Nt < 100000 or Nt > self.Nmax:
            penalty += (min(abs(Nt - 100000), abs(Nt - self.Nmax))) ** 2

        # Tax rate constraints
        if tau_t < 0.02 or tau_t > 0.08:
            penalty += (min(abs(tau_t - 0.02), abs(tau_t - 0.08))) ** 2

        # Investment ratio constraints
        total_inv = k5 + k6 + k7
        if total_inv < 0.15 or total_inv > 0.4:
            penalty += (min(abs(total_inv - 0.15), abs(total_inv - 0.4))) ** 2

        # Environmental constraints
        co2_emissions = Nt * self.CO2p
        if co2_emissions > self.CO2max:
            penalty += (co2_emissions - self.CO2max) ** 2

        return penalty * self.penalty_coefficient

    def objective(self, x):
        """Enhanced objective function with penalty term"""
        Nt, tau_t, k5, k6, k7 = x

        # Calculate base objective components
        Re = self.Pt * Nt / self.Restandrand

        P = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        Cbase = self.Cbase + self.alpha1 * P[2]
        Cwaste = self.Cwaste + self.alpha2 * P[0]
        Cwater = self.Cwater + self.alpha2 * P[1]

        E = (
            self.k1 * (self.CO2p * Nt - Cbase) / self.CO2max
            + self.k2 * Nt / Cwaste
            + self.k3 * Nt / Cwater
        )

        Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = Sresidents / self.Sbase

        # Add penalty term to objective
        penalty = self.constraint_violation_penalty(x)

        return -(Re + S - E) + penalty

    def multi_start_optimize(self, n_starts=10):
        """Perform multi-start optimization"""
        best_result = None
        best_objective = float("inf")

        # Define different starting points
        Nt_starts = np.linspace(100000, self.Nmax * 0.8, n_starts)
        tau_starts = np.linspace(0.02, 0.07, n_starts)
        k_starts = np.linspace(0.05, 0.15, n_starts)

        for i in range(n_starts):
            x0 = [Nt_starts[i], tau_starts[i], k_starts[i], k_starts[i], k_starts[i]]

            bounds = [
                (100000, self.Nmax),
                (0.02, 0.08),
                (0.05, 0.2),
                (0.05, 0.2),
                (0.05, 0.2),
            ]

            result = minimize(
                self.objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 10000, "ftol": 1e-9, "disp": False, "eps": 1e-8},
            )

            if result.success and result.fun < best_objective:
                best_objective = result.fun
                best_result = deepcopy(result)

        return best_result

    def analyze_solution(self, result):
        """Analyze and print optimization results"""
        if result is None:
            print("No valid solution found.")
            return

        Nt, tau_t, k5, k6, k7 = result.x

        print("\nOptimization Results:")
        print(f"Number of tourists (Nt): {Nt:.0f}")
        print(f"Tourist tax rate (tau_t): {tau_t*100:.2f}%")
        print(f"Waste management investment ratio (k5): {k5:.3f}")
        print(f"Water management investment ratio (k6): {k6:.3f}")
        print(f"Environmental protection investment ratio (k7): {k7:.3f}")
        print(f"Total investment ratio: {(k5+k6+k7):.3f}")

        # Calculate and print key metrics
        Re = self.Pt * Nt
        P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        print("\nKey Metrics:")
        print(f"Total Revenue: ${Re:,.2f}")
        print(f"CO2 Emissions: {Nt * self.CO2p:.2f}")
        print(f"Waste Management Investment: ${P_waste:,.2f}")
        print(f"Water Management Investment: ${P_water:,.2f}")
        print(
            f"Resident Satisfaction: {(self.a1 * Nt**2 + self.a2 * Nt + self.b1):.2f}"
        )

        print(f"\nObjective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")

    def sensitivity_analysis(self, n_samples=1000):
        """Sobol全局敏感性分析"""
        # 定义分析参数及其范围
        problem = {
            "num_vars": 5,
            "names": ["Nt", "tau_t", "k5", "k6", "k7"],
            "bounds": [
                [100_000, self.Nmax],  # Nt范围
                [0.02, 0.08],  # tau_t范围
                [0.05, 0.2],
                [0.05, 0.2],
                [0.05, 0.2],  # k5, k6, k7范围
            ],
        }

        # 生成参数样本（Saltelli采样）
        param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

        # 运行模型并获取目标函数值
        outputs = []
        for params in param_values:
            # 将参数代入目标函数（需确保参数在合法范围内）
            try:
                z = -self.objective(params)  # 目标函数值Z = Re + S - E
                outputs.append(z)
            except:
                outputs.append(np.nan)

        # 分析敏感性指数
        si = sobol.analyze(problem, np.array(outputs), calc_second_order=True)

        # 输出结果
        print("\nSobol敏感性分析结果:")
        print("一阶指数 (S_i):")
        for name, s in zip(problem["names"], si["S1"]):
            print(f"{name}: {s:.3f}")
        print("\n总阶指数 (S_Ti):")
        for name, s in zip(problem["names"], si["ST"]):
            print(f"{name}: {s:.3f}")

        # 可视化
        self.plot_sensitivity(si, problem["names"])

    def plot_sensitivity(self, si, names):
        """绘制敏感性指数图"""
        plt.figure(figsize=(10, 6))
        plt.bar(names, si["S1"], label="一阶指数 (S_i)", alpha=0.7)
        plt.bar(
            names, si["ST"] - si["S1"], bottom=si["S1"], label="高阶交互作用", alpha=0.7
        )
        plt.xlabel("参数")
        plt.ylabel("敏感性指数")
        plt.title("Sobol全局敏感性分析")
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    optimizer = ImprovedTourismOptimizer()
    result = optimizer.multi_start_optimize(n_starts=20)
    optimizer.analyze_solution(result)

    optimizer.sensitivity_analysis(n_samples=500)


if __name__ == "__main__":
    main()
