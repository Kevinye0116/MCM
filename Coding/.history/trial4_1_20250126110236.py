import numpy as np
from scipy.optimize import minimize


class TourismOptimizer:
    def __init__(self):
        # Constants
        self.Nmax = 2000000  # Maximum tourists per day
        self.CO2max = 500000  # Maximum CO2 emissions
        self.Cbase = 50
        self.Cwaste = 18900000  # Waste capacity
        self.Cwater = 2549110  # Water capacity
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

        # Add time-related parameters
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
        # Initialize capacity arrays
        C_waste = np.zeros(len(self.time))
        C_water = np.zeros(len(self.time))
        C_base = np.zeros(len(self.time))

        # Set initial conditions
        C_waste[0] = self.Cwaste
        C_water[0] = self.Cwater
        C_base[0] = self.Cbase

        # Calculate investments
        P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        # Time integration using Euler method
        for i in range(1, len(self.time)):
            # Update capacities according to equation (4)
            C_waste[i] = C_waste[i - 1] + self.alpha1 * P_waste * self.dt
            C_water[i] = C_water[i - 1] + self.alpha2 * P_water * self.dt
            C_base[i] = C_base[i - 1] + self.alpha3 * P_e * self.dt

        return C_waste[-1], C_water[-1], C_base[-1]

    def objective(self, x):
        """
        Objective function to maximize Z = Re + S - E
        x = [Nt, tau_t, k5, k6, k7]
        """
        Nt, tau_t, k5, k6, k7 = x

        # Simulate capacity evolution
        C_waste_final, C_water_final, C_base_final = self.simulate_capacity_evolution(
            Nt, tau_t, k5, k6, k7
        )

        # Calculate revenue (equation 1)
        Re = self.Pt * Nt

        # Calculate environmental impact (equation 2)
        E = (
            self.k1 * (self.CO2p * Nt - C_base_final)
            + self.k2 * Nt / C_waste_final
            + self.k3 * Nt / C_water_final
        )

        # Calculate societal satisfaction (equation 3)
        Sresidents = self.a1 * Nt**2 + self.a2 * Nt + self.b1
        S = self.k4 * Sresidents

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
            },  # Total investment ratio <= 0.4
            {
                "type": "ineq",
                "fun": lambda x: x[2] + x[3] + x[4] - 0.15,
            },  # Total investment ratio >= 0.15
            # Individual investment ratios
            {"type": "ineq", "fun": lambda x: x[2] - 0.05},  # k5 >= 0.05
            {"type": "ineq", "fun": lambda x: x[3] - 0.05},  # k6 >= 0.05
            {"type": "ineq", "fun": lambda x: x[4] - 0.05},  # k7 >= 0.05
        ]
        return cons

    def optimize(self):
        """Optimization function"""
        # Initial guess
        x0 = [500000, 0.04, 0.1, 0.1, 0.1]  # [Nt, tau_t, k5, k6, k7]

        bounds = [
            (100000, self.Nmax),  # Nt: reasonable lower bound
            (0.01, 0.08),  # tau_t: reasonable upper bound
            (0.05, 0.2),  # k5: narrow investment ratio range
            (0.05, 0.2),  # k6
            (0.05, 0.2),  # k7
        ]

        # Optimization using SLSQP method
        result = minimize(
            self.objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=self.constraints(),
            options={
                "maxiter": 2000,  # Increase max iterations
                "ftol": 1e-8,  # Increase precision
                "disp": True,  # Display optimization process
                "eps": 1e-10,  # Decrease gradient step size
            },
        )

        return result

    def analyze_solution(self, result):
        """Analyze and print optimization results"""
        Nt, tau_t, k5, k6, k7 = result.x

        # Calculate final capacities
        C_waste_final, C_water_final, C_base_final = self.simulate_capacity_evolution(
            Nt, tau_t, k5, k6, k7
        )

        print("\nOptimization Results:")
        print(f"Number of tourists (Nt): {Nt:.0f}")
        print(f"Tourist tax rate (tau_t): {tau_t*100:.2f}%")
        print(f"Waste management investment ratio (k5): {k5:.3f}")
        print(f"Water management investment ratio (k6): {k6:.3f}")
        print(f"Environmental protection investment ratio (k7): {k7:.3f}")
        print(f"Total investment ratio: {(k5+k6+k7):.3f}")
        print("\nFinal Capacities:")
        print(f"Waste capacity: {C_waste_final:.0f}")
        print(f"Water capacity: {C_water_final:.0f}")
        print(f"Base environmental capacity: {C_base_final:.0f}")
        print(f"\nObjective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")


if __name__ == "__main__":
    optimizer = TourismOptimizer()
    result = optimizer.optimize()
    optimizer.analyze_solution(result)
