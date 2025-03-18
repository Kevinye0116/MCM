from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


class TourismOptimization:
    def __init__(self):
        # Model parameters
        self.k1 = 0.1  # Environmental protection effectiveness
        self.k2 = 0.05  # Carbon emission impact
        self.k3 = 0.2  # Nature recovery rate
        self.k4 = 0.15  # Infrastructure development rate
        self.k5 = 0.3  # Infrastructure investment coefficient
        self.k6 = 0.25  # Environmental protection investment coefficient

        # Constants
        self.a1 = 0.01  # Resident satisfaction coefficient
        self.a2 = 0.02  # Tourist tax impact
        self.a3 = 0.015  # Tourist spending impact
        self.b1 = 100  # Base resident satisfaction
        self.b2 = 100  # Base tourist satisfaction
        self.gamma = 0.1  # Tourist growth rate
        self.delta = 0.05  # Environmental impact factor

        # Constraints
        self.Nt_max = 10000  # Maximum tourists
        self.tau_max = 100  # Maximum tourist tax
        self.Pt_max = 1000  # Maximum tourist spending
        self.CO2_max = 500000  # Maximum CO2 emissions
        self.E_min = 50  # Minimum environmental quality

        # Hidden cost and other parameters
        self.Ch = 1000  # Hidden cost
        self.CO2p = 0.5  # CO2 per person

    def calculate_revenue(self, x: np.ndarray) -> float:
        """Calculate total revenue (objective 1)"""
        Nt, tau_t, Pt = x
        return (tau_t + Pt) * Nt - self.k6 * tau_t * Nt - self.k5 * tau_t * Nt - self.Ch

    def calculate_social_satisfaction(self, x: np.ndarray) -> float:
        """Calculate social satisfaction (objective 2)"""
        Nt, tau_t, Pt = x
        S_residents = -self.a1 * Nt + self.b1
        S_tourists = -self.a2 * tau_t - self.a3 * Pt + self.b2
        return S_residents + S_tourists

    def calculate_environmental_impact(self, x: np.ndarray) -> float:
        """Calculate environmental quality index (objective 3)"""
        Nt, tau_t, Pt = x
        # Environmental quality index E increases with environmental protection investment
        # and decreases with tourist numbers (CO2 emissions)
        E = self.k1 * self.k6 * tau_t * Nt - self.k2 * self.CO2p * Nt + self.k3 * 100
        return E  # Higher E means better environmental quality

    def constraints(self, x: np.ndarray) -> List[float]:
        """Define all constraints"""
        Nt, tau_t, Pt = x
        constraints = [
            self.Nt_max - Nt,  # Tourism capacity constraint
            self.tau_max - tau_t,  # Tax constraint
            self.Pt_max - Pt,  # Spending constraint
            self.CO2_max - Nt * self.CO2p,  # Environmental constraint
            -self.a1 * Nt + self.b1 - 60,  # Societal constraint
            Nt,  # Non-negative tourists
            tau_t,  # Non-negative tax
            Pt,  # Non-negative spending
        ]
        return constraints

    def pareto_optimization(
        self, population_size: int = 100, generations: int = 50
    ) -> List[np.ndarray]:
        """Perform Pareto optimization"""
        population = []
        for _ in range(population_size):
            # Generate random initial solutions within bounds
            x = np.array(
                [
                    np.random.uniform(0, self.Nt_max),
                    np.random.uniform(0, self.tau_max),
                    np.random.uniform(0, self.Pt_max),
                ]
            )
            population.append(x)

        pareto_front = []
        for generation in range(generations):
            # Evaluate objectives for current population
            objectives = []
            for solution in population:
                if all(c >= 0 for c in self.constraints(solution)):
                    obj1 = self.calculate_revenue(solution)
                    obj2 = self.calculate_social_satisfaction(solution)
                    obj3 = self.calculate_environmental_impact(solution)
                    objectives.append((obj1, obj2, obj3))
                else:
                    objectives.append((-np.inf, -np.inf, -np.inf))

            # Update Pareto front
            for i, solution in enumerate(population):
                dominated = False
                for j, other_solution in enumerate(population):
                    if i != j and self.dominates(objectives[j], objectives[i]):
                        dominated = True
                        break
                if not dominated and all(c >= 0 for c in self.constraints(solution)):
                    pareto_front.append(solution)

            # Generate new population
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament = np.random.choice(len(population), 2, replace=False)
                if objectives[tournament[0]][0] > objectives[tournament[1]][0]:
                    parent1 = population[tournament[0]]
                else:
                    parent1 = population[tournament[1]]

                tournament = np.random.choice(len(population), 2, replace=False)
                if objectives[tournament[0]][1] > objectives[tournament[1]][1]:
                    parent2 = population[tournament[0]]
                else:
                    parent2 = population[tournament[1]]

                # Crossover and mutation
                child = self.crossover_and_mutate(parent1, parent2)
                new_population.append(child)

            population = new_population

        return pareto_front

    def dominates(
        self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]
    ) -> bool:
        """Check if obj1 dominates obj2"""
        return (
            obj1[0] >= obj2[0]
            and obj1[1] >= obj2[1]
            and obj1[2] >= obj2[2]
            and (obj1[0] > obj2[0] or obj1[1] > obj2[1] or obj1[2] > obj2[2])
        )

    def crossover_and_mutate(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """Perform crossover and mutation"""
        # Crossover
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2

        # Mutation
        mutation_rate = 0.1
        if np.random.random() < mutation_rate:
            mutation_strength = 0.1
            child += np.random.normal(0, mutation_strength, size=3)

        # Ensure bounds
        child[0] = np.clip(child[0], 0, self.Nt_max)
        child[1] = np.clip(child[1], 0, self.tau_max)
        child[2] = np.clip(child[2], 0, self.Pt_max)

        return child

    def visualize_pareto_front(self, pareto_front: List[np.ndarray]):
        """Visualize the Pareto front in 3D"""
        objectives = np.array(
            [
                [
                    self.calculate_revenue(x),
                    self.calculate_social_satisfaction(x),
                    self.calculate_environmental_impact(x),
                ]
                for x in pareto_front
            ]
        )

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            objectives[:, 0], objectives[:, 1], objectives[:, 2], c="b", marker="o"
        )

        ax.set_xlabel("Revenue")
        ax.set_ylabel("Social Satisfaction")
        ax.set_zlabel("Environmental Impact")
        ax.set_title("Pareto Front")

        plt.show()


# Usage example
def main():
    optimizer = TourismOptimization()
    pareto_front = optimizer.pareto_optimization(population_size=100, generations=50)

    print("\nPareto optimal solutions:")
    for solution in pareto_front[:5]:  # Show first 5 solutions
        Nt, tau_t, Pt = solution
        revenue = optimizer.calculate_revenue(solution)
        satisfaction = optimizer.calculate_social_satisfaction(solution)
        environment = optimizer.calculate_environmental_impact(solution)

        print(f"\nTourists: {Nt:.0f}")
        print(f"Tourist tax: {tau_t:.2f}")
        print(f"Tourist spending: {Pt:.2f}")
        print(f"Revenue: {revenue:.2f}")
        print(f"Social satisfaction: {satisfaction:.2f}")
        print(f"Environmental impact: {environment:.2f}")

    optimizer.visualize_pareto_front(pareto_front)


if __name__ == "__main__":
    np.random.seed(42)
    main()
