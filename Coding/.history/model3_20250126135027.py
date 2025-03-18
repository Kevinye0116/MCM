import time

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize


class HybridDETourismOptimizer:
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

        # Optimization parameters
        self.bounds = [
            (100000, self.Nmax),  # Nt
            (0.02, 0.08),  # tau_t
            (0.05, 0.2),  # k5
            (0.05, 0.2),  # k6
            (0.05, 0.2),  # k7
        ]

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def objective(self, x):
        """Objective function"""
        Nt, tau_t, k5, k6, k7 = x

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

        return -(Re + S - E)

    def constraint_functions(self, x):
        """Define all constraint functions"""
        Nt, tau_t, k5, k6, k7 = x

        # Investment ratio constraints
        total_inv = k5 + k6 + k7
        inv_lower = total_inv - 0.15
        inv_upper = 0.4 - total_inv

        # Environmental constraints
        co2_constraint = self.CO2max - Nt * self.CO2p

        # Infrastructure constraints
        waste_constraint = self.Cwaste * 1.2 / 365 - 0.012 * Nt
        water_constraint = self.Cwater * 1.2 / 365 - 0.012 * Nt

        return np.array(
            [inv_lower, inv_upper, co2_constraint, waste_constraint, water_constraint]
        )

    def create_nonlinear_constraints(self):
        """Create NonlinearConstraint objects"""
        # Define bounds for each constraint
        lower_bounds = [0, -np.inf, 0, 0, 0]  # All constraints should be >= 0
        upper_bounds = [np.inf, 0, np.inf, np.inf, np.inf]  # Upper bounds where needed

        return NonlinearConstraint(
            self.constraint_functions, lower_bounds, upper_bounds
        )

    def hybrid_optimize(self, de_popsize=20, de_maxiter=50, local_maxiter=1000):
        """Hybrid optimization using DE and SLSQP"""
        print("Starting Differential Evolution optimization...")

        # First phase: Differential Evolution for global search
        de_result = differential_evolution(
            self.objective,
            bounds=self.bounds,
            constraints=self.create_nonlinear_constraints(),
            popsize=de_popsize,
            maxiter=de_maxiter,
            polish=False,  # Disable automatic local optimization
            updating="deferred",
            workers=-1,  # Use all available CPU cores
            disp=True,
        )

        if not de_result.success:
            print("Warning: Differential Evolution did not converge")

        print("\nStarting local optimization with SLSQP...")

        # Second phase: SLSQP for local refinement
        final_result = minimize(
            self.objective,
            de_result.x,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.create_nonlinear_constraints(),
            options={"maxiter": local_maxiter, "ftol": 1e-9, "disp": True},
        )

        return final_result

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

        # Calculate key metrics
        Re = self.Pt * Nt
        P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        print("\nKey Metrics:")
        print(f"Total Revenue: ${Re:,.2f}")
        print(f"CO2 Emissions: {Nt * self.CO2p:.2f}")
        print(
            f"Resident Satisfaction: {(self.a1 * Nt**2 + self.a2 * Nt + self.b1):.2f}"
        )

        # Check constraint violations
        constraints = self.constraint_functions(result.x)
        print("\nConstraint Values:")
        print(f"Investment ratio constraints: {constraints[0:2]}")
        print(f"CO2 constraint: {constraints[2]}")
        print(f"Infrastructure constraints: {constraints[3:5]}")

        print(f"\nFinal objective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")


def main():
    start_time = time.time()

    optimizer = HybridDETourismOptimizer()
    result = optimizer.hybrid_optimize(
        de_popsize=50,  # Population size for DE
        de_maxiter=120,  # Maximum iterations for DE
        local_maxiter=10000,  # Maximum iterations for local optimization
    )
    optimizer.analyze_solution(result)

    end_time = time.time()
    print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
