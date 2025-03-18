import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


class TourismOptimizer:
    def __init__(self):
        # Constants
        self.Nmax = 2000000  # Maximum tourists per day
        self.CO2max = 500000  # Maximum CO2 emissions
        self.Cbase = 50
        self.Cwaste = 18900000  # Initial waste capacity
        self.Cwater = 2549110  # Initial water capacity
        self.Pt = 190  # Average spending per tourist
        self.CO2p = 0.184  # Carbon emissions per person
        self.Ptmax = 200  # Maximum infrastructure investment

        # Coefficients
        self.k1 = 0.5  # Environmental impact weights
        self.k2 = 0.3
        self.k3 = 0.2
        self.k4 = 1.0  # Societal satisfaction scaling
        self.a1 = -1.975284171471327e-11  # Resident satisfaction quadratic term
        self.a2 = 4.772497706879391e-05  # Resident satisfaction linear term
        self.b1 = 39.266042633247565  # Resident satisfaction constant
        self.alpha1 = 0.2  # Waste capacity growth rate
        self.alpha2 = 0.05  # Water capacity growth rate
        self.alpha3 = 0.05  # Environmental protection effectiveness

        # Time-related parameters
        self.dt = 1  # Time step (1 day)
        self.T = 365  # Simulation period (1 year)
        self.time = np.arange(0, self.T, self.dt)

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations based on revenue"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def simulate_capacity_evolution(self, Nt, tau_t, k5, k6, k7):
        """Simulate the evolution of capacities over time"""
        C_waste = np.zeros(len(self.time))
        C_water = np.zeros(len(self.time))
        C_base = np.zeros(len(self.time))

        # Initial conditions
        C_waste[0] = self.Cwaste
        C_water[0] = self.Cwater
        C_base[0] = self.Cbase

        # Convert annual tourists to daily average
        Nt_daily = Nt / 365

        # Calculate daily investments based on daily tourist numbers
        P_waste, P_water, P_e = self.calculate_investments(Nt_daily, tau_t, k5, k6, k7)

        # Euler integration
        for i in range(1, len(self.time)):
            C_waste[i] = C_waste[i - 1] + self.alpha1 * P_waste * self.dt
            C_water[i] = C_water[i - 1] + self.alpha2 * P_water * self.dt
            C_base[i] = C_base[i - 1] + self.alpha3 * P_e * self.dt

        return C_waste[-1], C_water[-1], C_base[-1]

    def objective(self, x):
        """Objective function to maximize Z = Re + S - E"""
        Nt, tau_t, k5, k6, k7 = x

        # Simulate capacity evolution
        C_waste_final, C_water_final, C_base_final = self.simulate_capacity_evolution(
            Nt, tau_t, k5, k6, k7
        )

        # Calculate components
        Re = self.Pt * Nt
        E = (
            self.k1 * (self.CO2p * Nt - C_base_final)
            + self.k2 * Nt / C_waste_final
            + self.k3 * Nt / C_water_final
        )
        Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = self.k4 * Sresidents

        return -(Re + S - E)  # Minimize negative of Z

    def constraints(self):
        """Define optimization constraints"""
        cons = [
            # Financial constraints
            {"type": "ineq", "fun": lambda x: 0.08 - x[1]},  # τ_t ≤8%
            # Tourism constraints
            {"type": "ineq", "fun": lambda x: self.Nmax - x[0]},  # Nt ≤Nmax
            {"type": "ineq", "fun": lambda x: x[0] - 100000},  # Nt ≥1e5
            # Environmental constraints
            {"type": "ineq", "fun": lambda x: self.CO2max - x[0] * self.CO2p},  # CO2
            # Waste and water capacity constraints
            {
                "type": "ineq",
                "fun": lambda x: self.simulate_capacity_evolution(
                    x[0], x[1], x[2], x[3], x[4]
                )[0]
                - 3.65 * x[0],
            },  # C_waste_final ≥3.65*Nt
            {
                "type": "ineq",
                "fun": lambda x: self.simulate_capacity_evolution(
                    x[0], x[1], x[2], x[3], x[4]
                )[1]
                - 3.65 * x[0],
            },  # C_water_final ≥3.65*Nt
            # Societal constraint
            {
                "type": "ineq",
                "fun": lambda x: (self.a1 * x[0] ** 2 + self.a2 * x[0] + self.b1) - 60,
            },  # S_residents ≥60
            # Investment constraints
            {"type": "ineq", "fun": lambda x: 0.4 - (x[2] + x[3] + x[4])},  # Σk ≤0.4
            {"type": "ineq", "fun": lambda x: x[2]},  # k5 ≥0
            {"type": "ineq", "fun": lambda x: x[3]},  # k6 ≥0
            {"type": "ineq", "fun": lambda x: x[4]},  # k7 ≥0
        ]
        return cons

    def optimize(self):
        """Perform optimization"""
        # Initial guess
        x0 = [500000, 0.04, 0.1, 0.1, 0.1]  # [Nt, tau_t, k5, k6, k7]

        # Bounds
        bounds = [
            (100000, self.Nmax),  # Nt
            (0.0, 0.08),  # τ_t (0-8%)
            (0.0, None),  # k5 (≥0)
            (0.0, None),  # k6
            (0.0, None),  # k7
        ]

        # Optimize
        result = minimize(
            self.objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=self.constraints(),
            options={"maxiter": 2000, "ftol": 1e-8, "disp": True, "eps": 1e-10},
        )

        # Retry with alternative initial guess if failed
        if not result.success:
            print("\nRetrying with alternative initial guess...")
            x0_alt = [300000, 0.03, 0.1, 0.1, 0.2]
            result = minimize(
                self.objective,
                x0_alt,
                method="SLSQP",
                bounds=bounds,
                constraints=self.constraints(),
                options={"maxiter": 2000, "ftol": 1e-8, "disp": True},
            )

        return result

    def analyze_solution(self, result):
        """Analyze and print optimization results"""
        Nt, tau_t, k5, k6, k7 = result.x

        # Calculate final capacities
        C_waste, C_water, C_base = self.simulate_capacity_evolution(
            Nt, tau_t, k5, k6, k7
        )

        print("\n=== Optimization Results ===")
        print(f"Tourists (Nt): {Nt:.0f}")
        print(f"Tax Rate (τ_t): {tau_t*100:.2f}%")
        print(f"Investment Ratios: k5={k5:.3f}, k6={k6:.3f}, k7={k7:.3f}")
        print(f"Total Investment: {k5+k6+k7:.3f} (≤0.4)")
        print(f"\nFinal Capacities:")
        print(f"Waste: {C_waste:.0f} (≥{3.65*Nt:.0f})")
        print(f"Water: {C_water:.0f} (≥{3.65*Nt:.0f})")
        print(f"Base Environment: {C_base:.0f}")

        # Calculate objective components
        Re = self.Pt * Nt
        E = (
            self.k1 * (self.CO2p * Nt - C_base)
            + self.k2 * Nt / C_waste
            + self.k3 * Nt / C_water
        )
        Sres = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = self.k4 * Sres
        Z = Re + S - E

        print("\n=== Objective Breakdown ===")
        print(f"Revenue (Re): {Re:.2f}")
        print(f"Environmental Impact (E): {E:.2f}")
        print(f"Social Satisfaction (S): {S:.2f}")
        print(f"Resident Satisfaction: {Sres:.2f} (≥60)")
        print(f"\nTotal Objective (Z): {Z:.2f}")
        print(f"\nOptimization Status: {result.success}")
        print(f"Message: {result.message}")


# Execute optimization
if __name__ == "__main__":
    optimizer = TourismOptimizer()
    result = optimizer.optimize()
    optimizer.analyze_solution(result)
