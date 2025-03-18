from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
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


class TourismOptimizerWithSensitivity(ImprovedTourismOptimizer):
    def perform_sensitivity_analysis(self):
        """Perform sensitivity analysis on key parameters"""
        print("\nPerforming Sensitivity Analysis...")

        # Baseline optimization
        baseline_result = self.multi_start_optimize(n_starts=10)
        baseline_obj = -baseline_result.fun if baseline_result else 0
        baseline_tourists = baseline_result.x[0] if baseline_result else 0

        # Parameters to analyze
        analyses = {
            "CO2 Limit": {
                "param": "CO2max",
                "base": self.CO2max,
                "variations": np.linspace(self.CO2max * 0.5, self.CO2max * 1.5, 5),
            },
            "Tourist Tax Max": {
                "param": "tax_max",
                "base": 0.08,
                "variations": [0.05, 0.06, 0.07, 0.08, 0.09],
            },
            "Investment Ratio Max": {
                "param": "inv_max",
                "base": 0.4,
                "variations": [0.3, 0.35, 0.4, 0.45, 0.5],
            },
        }

        sensitivity_results = {}

        for param_name, config in analyses.items():
            results = []
            tourists = []
            base_value = config["base"]

            for value in config["variations"]:
                # Temporarily modify parameter
                if config["param"] == "CO2max":
                    self.CO2max = value
                elif config["param"] == "tax_max":
                    self.bounds[1] = (0.02, value)
                elif config["param"] == "inv_max":
                    self.bounds[2:5] = [(0.05, value / 3)] * 3

                # Run optimization
                result = self.multi_start_optimize(n_starts=5)

                if result:
                    results.append(-result.fun)
                    tourists.append(result.x[0])
                else:
                    results.append(np.nan)
                    tourists.append(np.nan)

                # Reset parameter
                if config["param"] == "CO2max":
                    self.CO2max = base_value
                elif config["param"] == "tax_max":
                    self.bounds[1] = (0.02, 0.08)
                elif config["param"] == "inv_max":
                    self.bounds[2:5] = [(0.05, 0.2)] * 3

            sensitivity_results[param_name] = {
                "variations": config["variations"],
                "objective_values": results,
                "tourist_numbers": tourists,
            }

        self._plot_sensitivity_results(
            sensitivity_results, baseline_obj, baseline_tourists
        )
        self._calculate_elasticities(sensitivity_results)

        return sensitivity_results

    def _plot_sensitivity_results(self, results, baseline_obj, baseline_tourists):
        """Plot sensitivity analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Sensitivity Analysis Results", fontsize=16)

        for idx, (param_name, data) in enumerate(results.items()):
            # Plot objective value changes
            ax1 = axes[0, idx]
            variations = data["variations"]
            obj_values = data["objective_values"]

            ax1.plot(variations, obj_values, "b-o")
            ax1.axhline(y=baseline_obj, color="r", linestyle="--", label="Baseline")
            ax1.set_title(f"{param_name} - Objective Value")
            ax1.set_xlabel("Parameter Value")
            ax1.set_ylabel("Objective Value")
            ax1.grid(True)

            # Plot tourist number changes
            ax2 = axes[1, idx]
            tourist_numbers = data["tourist_numbers"]

            ax2.plot(variations, tourist_numbers, "g-o")
            ax2.axhline(
                y=baseline_tourists, color="r", linestyle="--", label="Baseline"
            )
            ax2.set_title(f"{param_name} - Tourist Numbers")
            ax2.set_xlabel("Parameter Value")
            ax2.set_ylabel("Number of Tourists")
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def _calculate_elasticities(self, results):
        """Calculate and print elasticities for each parameter"""
        print("\nParameter Elasticities:")

        for param_name, data in results.items():
            variations = np.array(data["variations"])
            obj_values = np.array(data["objective_values"])
            tourist_numbers = np.array(data["tourist_numbers"])

            # Calculate average elasticity
            base_idx = len(variations) // 2
            base_param = variations[base_idx]
            base_obj = obj_values[base_idx]
            base_tourists = tourist_numbers[base_idx]

            # Calculate elasticities for adjacent points
            param_changes = (variations[1:] - variations[:-1]) / variations[:-1]
            obj_changes = (obj_values[1:] - obj_values[:-1]) / obj_values[:-1]
            tourist_changes = (
                tourist_numbers[1:] - tourist_numbers[:-1]
            ) / tourist_numbers[:-1]

            obj_elasticity = np.mean(obj_changes / param_changes)
            tourist_elasticity = np.mean(tourist_changes / param_changes)

            print(f"\n{param_name}:")
            print(f"  Objective Value Elasticity: {obj_elasticity:.3f}")
            print(f"  Tourist Numbers Elasticity: {tourist_elasticity:.3f}")


def main():
    optimizer = TourismOptimizerWithSensitivity()

    # Regular optimization
    print("Performing baseline optimization...")
    result = optimizer.multi_start_optimize(n_starts=20)
    optimizer.analyze_solution(result)

    # Sensitivity analysis
    sensitivity_results = optimizer.perform_sensitivity_analysis()


if __name__ == "__main__":
    main()
