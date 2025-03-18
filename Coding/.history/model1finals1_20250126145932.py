from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


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
