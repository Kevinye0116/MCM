import time
from copy import deepcopy

import numpy as np
from scipy.optimize import differential_evolution, minimize


class EnhancedTourismOptimizer:
    def __init__(self):
        # Original model parameters
        self.Nmax = 3000000
        self.CO2max = 250000
        self.Cbase = 500
        self.Cwater = 18900000
        self.Cwaste = 2549110
        self.Pt = 190
        self.CO2p = 0.184
        self.Ptmax = 200
        self.Restandrand = 500000000
        self.Sbase = 100
        self.k1 = 0.5
        self.k2 = 0.3
        self.k3 = 0.2
        self.a1 = -1.975284171471327e-11
        self.a2 = 4.772497706879391e-05
        self.b1 = 39.266042633247565
        self.alpha1 = 0.02
        self.alpha2 = 0.05
        self.alpha3 = 0.05

        # Enhanced optimization parameters
        self.penalty_coefficient = 1e6
        self.bounds = [
            (100000, self.Nmax),
            (0.02, 0.08),
            (0.05, 0.2),
            (0.05, 0.2),
            (0.05, 0.2),
        ]

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def adaptive_penalty(self, x, iteration, max_iterations):
        """Calculate adaptive penalty based on iteration progress"""
        Nt, tau_t, k5, k6, k7 = x
        base_penalty = 0

        # Progressive penalty weights
        progress = iteration / max_iterations
        weight = 1 + 9 * progress  # Penalty weight increases from 1 to 10

        # Tourism constraints with adaptive weights
        if Nt < 100000 or Nt > self.Nmax:
            base_penalty += weight * (min(abs(Nt - 100000), abs(Nt - self.Nmax))) ** 2

        # Tax rate constraints
        if tau_t < 0.02 or tau_t > 0.08:
            base_penalty += weight * (min(abs(tau_t - 0.02), abs(tau_t - 0.08))) ** 2

        # Investment ratio constraints
        total_inv = k5 + k6 + k7
        if total_inv < 0.15 or total_inv > 0.4:
            base_penalty += (
                weight * (min(abs(total_inv - 0.15), abs(total_inv - 0.4))) ** 2
            )

        # Environmental constraints with higher weight
        co2_emissions = Nt * self.CO2p
        if co2_emissions > self.CO2max:
            base_penalty += weight * 1.5 * (co2_emissions - self.CO2max) ** 2

        return base_penalty * self.penalty_coefficient

    def objective(self, x, iteration=0, max_iterations=1):
        """Enhanced objective function with adaptive penalty"""
        Nt, tau_t, k5, k6, k7 = x

        # Base objective components
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

        # Add adaptive penalty
        penalty = self.adaptive_penalty(x, iteration, max_iterations)

        return -(Re + S - E) + penalty

    def hybrid_optimize(self, de_popsize=20, de_maxiter=30, local_maxiter=1000):
        """Hybrid optimization combining DE and local search"""
        print("Starting global optimization with Differential Evolution...")

        # Phase 1: Differential Evolution
        de_result = differential_evolution(
            lambda x: self.objective(x, 0, de_maxiter),
            self.bounds,
            popsize=de_popsize,
            maxiter=de_maxiter,
            strategy="best1bin",
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating="deferred",
            workers=-1,
            disp=True,
        )

        print("\nStarting local refinement with SLSQP...")

        # Phase 2: Local optimization with multiple starts
        best_result = None
        best_objective = float("inf")

        # Generate multiple starting points around DE solution
        base_point = de_result.x
        perturbations = [np.random.normal(0, 0.1, len(base_point)) for _ in range(5)]

        for i, pert in enumerate([np.zeros_like(base_point)] + perturbations):
            start_point = np.clip(
                base_point + pert,
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds],
            )

            result = minimize(
                lambda x: self.objective(x, local_maxiter - 1, local_maxiter),
                start_point,
                method="SLSQP",
                bounds=self.bounds,
                options={"maxiter": local_maxiter, "ftol": 1e-9, "disp": True},
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

        # Calculate metrics
        Re = self.Pt * Nt
        P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        print("\nPerformance Metrics:")
        print(f"Total Revenue: ${Re:,.2f}")
        print(f"CO2 Emissions: {Nt * self.CO2p:.2f}")
        print(
            f"Resident Satisfaction: {(self.a1 * Nt**2 + self.a2 * Nt + self.b1):.2f}"
        )

        # Constraint analysis
        co2_constraint = self.CO2max - (Nt * self.CO2p)
        total_inv = k5 + k6 + k7

        print("\nConstraint Analysis:")
        print(f"CO2 Margin: {co2_constraint:.2f}")
        print(f"Investment Ratio: {total_inv:.3f} (Should be between 0.15 and 0.4)")

        print(f"\nFinal objective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")


def main():
    start_time = time.time()

    optimizer = EnhancedTourismOptimizer()
    result = optimizer.hybrid_optimize(
        de_popsize=20,  # DE population size
        de_maxiter=30,  # DE iterations
        local_maxiter=1000,  # Local search iterations
    )
    optimizer.analyze_solution(result)

    end_time = time.time()
    print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
