import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from mpl_toolkits.mplot3d import Axes3D


class TourismOptimizer:
    def __init__(self):
        # Model Constants
        self.K1 = 0.4  # Environmental impact coefficient for carbon emissions
        self.K2 = 0.3  # Environmental impact coefficient for waste
        self.K3 = 0.3  # Environmental impact coefficient for water
        self.A1 = -1.97e-11  # Coefficient for quadratic term in resident satisfaction
        self.A2 = 4.77e-05  # Coefficient for linear term in resident satisfaction
        self.B1 = 39.27  # Base satisfaction level
        self.K4 = 0.4  # Social satisfaction coefficient
        self.ALPHA1 = 0.2  # Waste management efficiency
        self.ALPHA2 = 0.2  # Water management efficiency
        self.ALPHA3 = 0.2  # Environmental management efficiency
        self.K5 = 0.15  # Waste management investment ratio
        self.K6 = 0.15  # Water management investment ratio
        self.K7 = 0.1  # Environmental management investment ratio

        # Constraints
        self.N_MAX = 20000  # Maximum daily tourists
        self.P_MAX = 500  # Maximum spending per tourist
        self.CO2_MAX = 0.1  # Maximum carbon emissions per person
        self.C_WASTE_MAX = 100000  # Maximum waste capacity
        self.C_WATER_MAX = 100000  # Maximum water capacity

        # DEAP Setup
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP tools and operators"""
        # Create fitness class and individual class
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register genes
        self.toolbox.register("nt", self._create_nt)
        self.toolbox.register("pt", self._create_pt)
        self.toolbox.register("tt", self._create_tt)

        # Register individual and population
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (self.toolbox.nt, self.toolbox.pt, self.toolbox.tt),
            n=1,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selNSGA2)

    def _create_nt(self):
        """Create tourist number gene"""
        return random.uniform(0, self.N_MAX)

    def _create_pt(self):
        """Create tourist spending gene"""
        return random.uniform(0, self.P_MAX)

    def _create_tt(self):
        """Create tax rate gene"""
        return random.uniform(0, 0.08)

    def calculate_objectives(self, individual):
        """Calculate the three objectives: Revenue, Social Satisfaction, Environmental Impact"""
        Nt, Pt, tt = individual

        # Calculate tourism revenue (Re)
        Re = Pt * Nt

        # Calculate environmental quality index (E)
        CO2p = 0.05  # Assumed constant CO2 per person
        Cbase = 1000  # Base carbon treatment capacity

        E = (
            self.K1 * (CO2p * Nt - Cbase)
            + self.K2 * (Nt / self.C_WASTE_MAX)
            + self.K3 * (Nt / self.C_WATER_MAX)
        )

        # Calculate social satisfaction (S)
        Sresidents = self.A1 * Nt**2 + self.A2 * Nt + self.B1
        S = self.K4 * Sresidents

        return Re, S, E

    def check_constraints(self, individual):
        """Check if solution satisfies all constraints"""
        Nt, Pt, tt = individual
        Re = Pt * Nt

        # Financial constraints
        if Re <= 0 or tt > 0.08 or Pt > self.P_MAX:
            return False

        # Tourism constraints
        if Nt < 0 or Nt > self.N_MAX:
            return False

        # Environmental constraints
        CO2p = 0.05
        if Nt * CO2p > self.CO2_MAX:
            return False
        if 0.012 * Nt > (1.2 / 365) * self.C_WASTE_MAX:
            return False
        if 0.012 * Nt > (1.2 / 365) * self.C_WATER_MAX:
            return False

        # Societal constraints
        Sresidents = self.A1 * Nt**2 + self.A2 * Nt + self.B1
        if Sresidents < 60:
            return False

        return True

    def evaluate(self, individual):
        """Evaluate fitness of an individual"""
        if not self.check_constraints(individual):
            return -1000000, -1000000, 1000000
        return self.calculate_objectives(individual)

    def optimize(self, pop_size=100, ngen=50):
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

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            fits[:, 0],
            fits[:, 1],
            fits[:, 2],
            c=fits[:, 2],  # Color by environmental impact
            cmap="viridis",
            marker="o",
        )

        plt.colorbar(scatter, label="Environmental Impact (E)")
        ax.set_xlabel("Tourism Revenue (Re)")
        ax.set_ylabel("Social Satisfaction (S)")
        ax.set_zlabel("Environmental Impact (E)")
        plt.title("Pareto Front for Tourism Optimization")
        plt.show()

    def print_best_solutions(self, population, n=5):
        """Print the best n solutions"""
        best_solutions = tools.sortNondominated(population, len(population))[0]
        print(f"\nTop {n} Solutions (Nt, Pt, τt):")
        for i, ind in enumerate(best_solutions[:n]):
            print(f"\nSolution {i+1}:")
            print(f"Tourist Number: {ind[0]:.0f}")
            print(f"Spending per Tourist: ${ind[1]:.2f}")
            print(f"Tax Rate: {ind[2]*100:.1f}%")
            print(f"Objectives:")
            print(f"- Revenue: ${ind.fitness.values[0]:,.2f}")
            print(f"- Social Satisfaction: {ind.fitness.values[1]:.2f}")
            print(f"- Environmental Impact: {ind.fitness.values[2]:.2f}")


def main():
    # Create optimizer instance
    optimizer = TourismOptimizer()

    # Run optimization
    print("Starting optimization...")
    final_population = optimizer.optimize(pop_size=100, ngen=50)

    # Plot results
    optimizer.plot_pareto_front(final_population)

    # Print best solutions
    optimizer.print_best_solutions(final_population)


if __name__ == "__main__":
    main()
