import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from mpl_toolkits.mplot3d import Axes3D

# Define constants
K1 = 0.001  # Environmental impact coefficient for carbon emissions
K2 = 0.1  # Environmental impact coefficient for waste
K3 = 0.1  # Environmental impact coefficient for water
A1 = -0.0001  # Coefficient for quadratic term in resident satisfaction
A2 = 0.1  # Coefficient for linear term in resident satisfaction
B1 = 50  # Base satisfaction level
K4 = 1.2  # Social satisfaction coefficient
ALPHA1 = 0.2  # Waste management efficiency
ALPHA2 = 0.2  # Water management efficiency
ALPHA3 = 0.2  # Environmental management efficiency
K5 = 0.15  # Waste management investment ratio
K6 = 0.15  # Water management investment ratio
K7 = 0.1  # Environmental management investment ratio

# Problem Bounds
N_MAX = 20000  # Maximum daily tourists
P_MAX = 500  # Maximum spending per tourist
CO2_MAX = 0.1  # Maximum carbon emissions per person
C_WASTE_MAX = 100000  # Maximum waste capacity
C_WATER_MAX = 100000  # Maximum water capacity

# Create fitness class and individual class
creator.create(
    "FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0)
)  # Maximize Re and S, Minimize E
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize toolbox
toolbox = base.Toolbox()


# Define genes
def create_nt():
    return random.uniform(0, N_MAX)


def create_pt():
    return random.uniform(0, P_MAX)


def create_tt():
    return random.uniform(0, 0.08)  # Tax rate <= 8%


toolbox.register("nt", create_nt)
toolbox.register("pt", create_pt)
toolbox.register("tt", create_tt)

# Create individual and population
N_PARAMS = 3  # Number of parameters to optimize (Nt, Pt, τt)

toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.nt, toolbox.pt, toolbox.tt),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def calculate_objectives(individual):
    Nt, Pt, tt = individual

    # Calculate tourism revenue (Re)
    Re = Pt * Nt

    # Calculate environmental quality index (E)
    CO2p = 0.05  # Assumed constant CO2 per person
    Cbase = 1000  # Base carbon treatment capacity
    Cwaste = C_WASTE_MAX
    Cwater = C_WATER_MAX

    E = K1 * (CO2p * Nt - Cbase) + K2 * (Nt / Cwaste) + K3 * (Nt / Cwater)

    # Calculate social satisfaction (S)
    Sresidents = A1 * Nt**2 + A2 * Nt + B1
    S = K4 * Sresidents

    return Re, S, E


def check_constraints(individual):
    Nt, Pt, tt = individual

    # Financial constraints
    if Re <= 0 or tt > 0.08 or Pt > P_MAX:
        return False

    # Tourism constraints
    if Nt < 0 or Nt > N_MAX:
        return False

    # Environmental constraints
    CO2p = 0.05
    if Nt * CO2p > CO2_MAX:
        return False
    if 0.012 * Nt > (1.2 / 365) * C_WASTE_MAX:
        return False
    if 0.012 * Nt > (1.2 / 365) * C_WATER_MAX:
        return False

    # Societal constraints
    Sresidents = A1 * Nt**2 + A2 * Nt + B1
    if Sresidents < 60:
        return False

    return True


def evaluate(individual):
    if not check_constraints(individual):
        return -1000000, -1000000, 1000000  # Return poor fitness for invalid solutions
    return calculate_objectives(individual)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)


def run_optimization():
    pop = toolbox.population(n=100)
    NGEN = 50

    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run algorithm
    final_pop = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=100,
        lambda_=100,
        cxpb=0.7,
        mutpb=0.3,
        ngen=NGEN,
        stats=stats,
        halloffame=None,
    )

    return final_pop[0]


def plot_pareto_front(population):
    fits = np.array([ind.fitness.values for ind in population])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(fits[:, 0], fits[:, 1], fits[:, 2], c="b", marker="o")
    ax.set_xlabel("Tourism Revenue (Re)")
    ax.set_ylabel("Social Satisfaction (S)")
    ax.set_zlabel("Environmental Impact (E)")
    plt.title("Pareto Front")
    plt.show()


# Run optimization
if __name__ == "__main__":
    final_population = run_optimization()

    # Plot results
    plot_pareto_front(final_population)

    # Print best solutions
    best_solutions = tools.sortNondominated(final_population, len(final_population))[0]
    print("\nBest Solutions (Nt, Pt, τt):")
    for ind in best_solutions[:5]:
        print(f"Parameters: {ind}")
        print(
            f"Objectives: Revenue={ind.fitness.values[0]:.2f}, "
            f"Social={ind.fitness.values[1]:.2f}, "
            f"Environmental={ind.fitness.values[2]:.2f}\n"
        )
