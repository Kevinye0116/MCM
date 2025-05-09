import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deap import algorithms, base, creator, tools
from mpl_toolkits.mplot3d import Axes3D
from SALib.analyze import sobol
from SALib.sample import saltelli


class ParetoTourismOptimizer:
    def __init__(self):
        # Model Constants
        self.Nmax = 3000000
        self.CO2max = 12800 + 1400000
        # self.Cbase = 50000
        self.Cwater = 18900000
        self.Cwaste = 4880000
        self.Pt = 190
        self.CO2p = 0.93
        self.Ptmax = 200
        self.Restandrand = 400000000
        self.Sbase = 100
        self.k1 = 0.4
        self.k2 = 0.3
        self.k3 = 0.3
        self.a1 = -1.97e-11
        self.a2 = 4.77e-05
        self.b1 = 39.27
        self.alpha1 = 0.0003
        self.alpha2 = 0.0005
        self.alpha3 = 0.0005

        # DEAP Setup
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP tools and operators"""
        # Create fitness class and individual class
        creator.create(
            "FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0)
        )  # Maximize Re and S, Minimize E
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register genes
        self.toolbox.register("nt", self._create_nt)
        self.toolbox.register("tau_t", self._create_tau_t)
        self.toolbox.register("k5", self._create_k)
        self.toolbox.register("k6", self._create_k)
        self.toolbox.register("k7", self._create_k)

        # Register individual and population
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                self.toolbox.nt,
                self.toolbox.tau_t,
                self.toolbox.k5,
                self.toolbox.k6,
                self.toolbox.k7,
            ),
            n=1,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            eta=15.0,
            low=[100000, 0.02, 0.05, 0.05, 0.05],
            up=[self.Nmax, 0.08, 0.2, 0.2, 0.2],
        )
        self.toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            eta=20.0,
            low=[100000, 0.02, 0.05, 0.05, 0.05],
            up=[self.Nmax, 0.08, 0.2, 0.2, 0.2],
            indpb=0.2,
        )
        self.toolbox.register("select", tools.selNSGA2)

    def _create_nt(self):
        """Create tourist number gene"""
        return random.uniform(100000, self.Nmax)

    def _create_tau_t(self):
        """Create tax rate gene"""
        return random.uniform(0.02, 0.08)

    def _create_k(self):
        """Create investment ratio gene"""
        return random.uniform(0.05, 0.2)

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations based on revenue"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def check_constraints(self, individual):
        """Check if solution satisfies all constraints"""
        Nt, tau_t, k5, k6, k7 = individual

        # Tourism constraints
        if Nt < 100000 or Nt > self.Nmax:
            return False

        # Tax rate constraints
        if tau_t < 0.02 or tau_t > 0.08:
            return False

        # Investment ratio constraints
        total_inv = k5 + k6 + k7
        if total_inv < 0.15 or total_inv > 0.4:
            return False

        if k5 < 0.05 or k6 < 0.05 or k7 < 0.05:
            return False

        if k5 > 0.2 or k6 > 0.2 or k7 > 0.2:
            return False

        # Environmental constraints
        co2_emissions = Nt * self.CO2p
        if co2_emissions > self.CO2max:
            return False

        return True

    def evaluate(self, individual):
        """Calculate the three objectives: Revenue, Social Satisfaction, Environmental Impact"""
        if not self.check_constraints(individual):
            return -1000000, -1000000, 1000000

        Nt, tau_t, k5, k6, k7 = individual

        # Calculate Revenue (Re)
        Re = self.Pt * Nt / self.Restandrand

        # Calculate investments and updated capacities
        P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)
        # Cbase_new = self.Cbase + self.alpha1 * P_e
        Cwaste_new = self.Cwaste + self.alpha2 * P_waste
        Cwater_new = self.Cwater + self.alpha2 * P_water

        # Calculate Environmental Impact (E)
        E = (
            self.k1 * (self.CO2p * Nt) / self.CO2max
            + self.k2 * Nt / Cwaste_new
            + self.k3 * Nt / Cwater_new
        )

        # Calculate Social Satisfaction (S)
        Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = Sresidents / self.Sbase

        return Re, S, E

    def optimize(self, pop_size=100, ngen=100):
        """Run the optimization"""
        pop = self.toolbox.population(n=pop_size)

        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Run algorithm
        final_pop = algorithms.eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=pop_size,
            lambda_=pop_size,
            cxpb=0.7,
            mutpb=0.3,
            ngen=ngen,
            stats=stats,
            halloffame=None,
        )

        return final_pop[0]

    def plot_pareto_front(self, population):
        """Plot the 3D Pareto front"""
        fits = np.array([ind.fitness.values for ind in population])
        valid_fits = fits[~np.any(fits <= -999999, axis=1)]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            valid_fits[:, 0],
            valid_fits[:, 1],
            valid_fits[:, 2],
            c=valid_fits[:, 2],
            cmap="viridis",
            marker="o",
        )

        plt.colorbar(scatter, label="Environmental Impact (E)")
        ax.set_xlabel("Normalized Revenue (Re)")
        ax.set_ylabel("Normalized Social Satisfaction (S)")
        ax.set_zlabel("Environmental Impact (E)")
        plt.title("Pareto Front for Tourism Optimization")
        plt.show()

    def print_best_solutions(self, population, n=20):
        """Print the best n solutions"""
        fronts = tools.sortNondominated(population, len(population))
        pareto_front = fronts[0]

        # Sort solutions by revenue (first objective)
        pareto_front.sort(key=lambda x: x.fitness.values[0], reverse=True)

        print(f"\nTop {n} Solutions from Pareto Front:")
        for i, ind in enumerate(pareto_front[:n]):
            Nt, tau_t, k5, k6, k7 = ind

            print(f"\nSolution {i+1}:")
            print(f"Tourist Number (Nt): {Nt:,.0f}")
            print(f"Tax Rate (τt): {tau_t*100:.1f}%")
            print(f"Waste Management Ratio (k5): {k5:.3f}")
            print(f"Water Management Ratio (k6): {k6:.3f}")
            print(f"Environmental Management Ratio (k7): {k7:.3f}")
            print(f"Total Investment Ratio: {(k5+k6+k7):.3f}")

            # Calculate actual values
            Re = self.Pt * Nt
            P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)
            Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1

            print("\nKey Metrics:")
            print(f"Total Revenue: ${Re:,.2f}")
            print(f"Normalized Revenue: {ind.fitness.values[0]:.3f}")
            print(f"Social Satisfaction: {ind.fitness.values[1]:.3f}")
            print(f"Environmental Impact: {ind.fitness.values[2]:.3f}")
            print(f"CO2 Emissions: {Nt * self.CO2p:,.2f}")
            print(f"Waste Management Investment: ${P_waste:,.2f}")
            print(f"Water Management Investment: ${P_water:,.2f}")
            print(f"Environmental Investment: ${P_e:,.2f}")

    def perform_sensitivity_analysis(self, num_samples=1024):
        """Perform Sobol sensitivity analysis"""
        # Define the problem
        problem = {
            "num_vars": 5,
            "names": ["Nt", "tau_t", "k5", "k6", "k7"],
            "bounds": [
                [100000, self.Nmax],  # Nt bounds
                [0.02, 0.08],  # tau_t bounds
                [0.05, 0.2],  # k5 bounds
                [0.05, 0.2],  # k6 bounds
                [0.05, 0.2],  # k7 bounds
            ],
        }

        # Generate samples
        param_values = saltelli.sample(problem, num_samples)

        # Run model for all samples
        Y_revenue = np.zeros([param_values.shape[0]])
        Y_satisfaction = np.zeros([param_values.shape[0]])
        Y_environment = np.zeros([param_values.shape[0]])

        for i, params in enumerate(param_values):
            results = self.evaluate(params)
            Y_revenue[i] = results[0]
            Y_satisfaction[i] = results[1]
            Y_environment[i] = results[2]

        # Perform Sobol analysis
        Si_revenue = sobol.analyze(problem, Y_revenue)
        Si_satisfaction = sobol.analyze(problem, Y_satisfaction)
        Si_environment = sobol.analyze(problem, Y_environment)

        return Si_revenue, Si_satisfaction, Si_environment

    def plot_sensitivity_analysis(self, Si_revenue, Si_satisfaction, Si_environment):
        """Plot sensitivity analysis results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        variables = ["Nt", "τt", "k5", "k6", "k7"]

        # Plot for Revenue
        indices = pd.DataFrame(
            {
                "S1": Si_revenue["S1"],
                "ST": Si_revenue["ST"],
            },
            index=variables,
        )
        indices.plot(kind="bar", ax=ax1)
        ax1.set_title("Revenue Sensitivity")
        ax1.set_ylabel("Sensitivity Index")

        # Plot for Social Satisfaction
        indices = pd.DataFrame(
            {
                "S1": Si_satisfaction["S1"],
                "ST": Si_satisfaction["ST"],
            },
            index=variables,
        )
        indices.plot(kind="bar", ax=ax2)
        ax2.set_title("Social Satisfaction Sensitivity")

        # Plot for Environmental Impact
        indices = pd.DataFrame(
            {
                "S1": Si_environment["S1"],
                "ST": Si_environment["ST"],
            },
            index=variables,
        )
        indices.plot(kind="bar", ax=ax3)
        ax3.set_title("Environmental Impact Sensitivity")

        plt.tight_layout()
        plt.show()

    def print_sensitivity_results(self, Si_revenue, Si_satisfaction, Si_environment):
        """Print detailed sensitivity analysis results"""
        variables = ["Nt", "τt", "k5", "k6", "k7"]

        print("\n=== Sensitivity Analysis Results ===")

        print("\n--- Revenue Sensitivity ---")
        print("First-order indices (S1):")
        for i, var in enumerate(variables):
            print(f"{var}: {Si_revenue['S1'][i]:.4f} ± {Si_revenue['S1_conf'][i]:.4f}")
        print("\nTotal-order indices (ST):")
        for i, var in enumerate(variables):
            print(f"{var}: {Si_revenue['ST'][i]:.4f} ± {Si_revenue['ST_conf'][i]:.4f}")

        print("\n--- Social Satisfaction Sensitivity ---")
        print("First-order indices (S1):")
        for i, var in enumerate(variables):
            print(
                f"{var}: {Si_satisfaction['S1'][i]:.4f} ± {Si_satisfaction['S1_conf'][i]:.4f}"
            )
        print("\nTotal-order indices (ST):")
        for i, var in enumerate(variables):
            print(
                f"{var}: {Si_satisfaction['ST'][i]:.4f} ± {Si_satisfaction['ST_conf'][i]:.4f}"
            )

        print("\n--- Environmental Impact Sensitivity ---")
        print("First-order indices (S1):")
        for i, var in enumerate(variables):
            print(
                f"{var}: {Si_environment['S1'][i]:.4f} ± {Si_environment['S1_conf'][i]:.4f}"
            )
        print("\nTotal-order indices (ST):")
        for i, var in enumerate(variables):
            print(
                f"{var}: {Si_environment['ST'][i]:.4f} ± {Si_environment['ST_conf'][i]:.4f}"
            )


def main():
    # Create optimizer instance
    optimizer = ParetoTourismOptimizer()

    # Run optimization
    print("Starting optimization...")
    final_population = optimizer.optimize(pop_size=200, ngen=100)

    # Plot results
    optimizer.plot_pareto_front(final_population)

    # Print best solutions
    optimizer.print_best_solutions(final_population)

    # Add sensitivity analysis
    print("\nPerforming sensitivity analysis...")
    Si_revenue, Si_satisfaction, Si_environment = (
        optimizer.perform_sensitivity_analysis()
    )

    # Print sensitivity analysis results
    optimizer.print_sensitivity_results(Si_revenue, Si_satisfaction, Si_environment)

    # Plot sensitivity analysis results
    print("\nPlotting sensitivity analysis results...")
    optimizer.plot_sensitivity_analysis(Si_revenue, Si_satisfaction, Si_environment)


if __name__ == "__main__":
    main()
