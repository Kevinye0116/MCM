import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class TourismOptimizer:
    def __init__(self):
        # Constants
        self.Nmax = 2000000  # Maximum tourists per day
        self.CO2max = 500000  # Maximum CO2 emissions
        self.Cbase = 50
        self.Cwaste = 1000  # Waste capacity
        self.Cwater = 1000  # Water capacity
        self.Pt = 375  # Average spending per tourist
        self.CO2p = 0.1  # Carbon emissions per person
        self.Ptmax = 1000  # Maximum infrastructure investment

        # Coefficients
        self.k1 = 0.3  # Environmental impact weights
        self.k2 = 0.3
        self.k3 = 0.4
        self.k4 = 1.0  # Societal satisfaction scaling
        self.a1 = -0.00001  # Resident satisfaction quadratic term
        self.a2 = 0.1  # Resident satisfaction linear term
        self.b1 = 50  # Resident satisfaction constant
        self.alpha1 = 0.1  # Waste capacity growth rate
        self.alpha2 = 0.1  # Water capacity growth rate
        self.alpha3 = 0.1  # Environmental protection effectiveness

    def objective(self, x):
        """
        Objective function to maximize Z = Re + S - E
        x = [Nt, tau_t, k5, k6, k7]
        """
        Nt, tau_t, k5, k6, k7 = x

        # Calculate revenue
        Re = self.Pt * Nt

        # Calculate environmental impact
        E = (
            self.k1 * (self.CO2p * Nt - self.Cbase)
            + self.k2 * Nt / self.Cwaste
            + self.k3 * Nt / self.Cwater
        )

        # Calculate societal satisfaction
        Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = self.k4 * Sresidents

        # Return negative of objective for minimization
        return -(Re + S - E)

    def constraints(self):
        """Define optimization constraints"""
        cons = [
            # Financial constraints
            {"type": "ineq", "fun": lambda x: x[1] - 0.08},  # tau_t <= 8%
            {"type": "ineq", "fun": lambda x: self.Ptmax - self.Pt},  # Pt <= Ptmax
            # Tourism constraints
            {"type": "ineq", "fun": lambda x: self.Nmax - x[0]},  # Nt <= Nmax
            {"type": "ineq", "fun": lambda x: x[0]},  # Nt >= 0
            # Environmental constraints
            {"type": "ineq", "fun": lambda x: self.CO2max - x[0] * self.CO2p},
            {"type": "ineq", "fun": lambda x: 1.2 / 365 * self.Cwaste - 0.012 * x[0]},
            {"type": "ineq", "fun": lambda x: 1.2 / 365 * self.Cwater - 0.012 * x[0]},
            # Societal constraints
            {
                "type": "ineq",
                "fun": lambda x: self.a1 * x[0] ** 2 + self.a2 * x[0] + self.b1 - 60,
            },
            # Investment ratio constraints
            {"type": "ineq", "fun": lambda x: 0.4 - (x[2] + x[3] + x[4])},
            {"type": "ineq", "fun": lambda x: x[2]},  # k5 >= 0
            {"type": "ineq", "fun": lambda x: x[3]},  # k6 >= 0
            {"type": "ineq", "fun": lambda x: x[4]},  # k7 >= 0
        ]
        return cons

    def optimize(self):
        # Initial guess
        x0 = [10000, 0.05, 0.1, 0.1, 0.1]

        # Bounds for variables
        bounds = [
            (0, self.Nmax),  # Nt
            (0, 0.08),  # tau_t
            (0, 0.4),  # k5
            (0, 0.4),  # k6
            (0, 0.4),  # k7
        ]

        # Optimize
        result = minimize(
            self.objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=self.constraints(),
            options={"maxiter": 1000},
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
        print(f"Objective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")


# Run optimization
if __name__ == "__main__":
    optimizer = TourismOptimizer()
    result = optimizer.optimize()
    optimizer.analyze_solution(result)
