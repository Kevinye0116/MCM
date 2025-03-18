import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools


# System Constants
class ModelConstants:
    def __init__(self):
        # Capacity constraints
        self.NTMAX = 20000  # Maximum daily tourists
        self.REMAX = 1000000  # Maximum daily revenue
        self.EMAX = 100  # Maximum environmental quality
        self.SMAX = 100  # Maximum social satisfaction
        self.CO2MAX = 100000  # Maximum CO2 emissions
        self.EMIN = 40  # Minimum environmental quality

        # Economic parameters
        self.c1 = 0.1  # Hidden cost linear coefficient
        self.c2 = 0.01  # Hidden cost quadratic coefficient

        # Environmental parameters
        self.k1 = 0.1  # Environmental protection effectiveness
        self.k2 = 0.001  # Environmental damage from CO2
        self.k3 = 0.05  # Natural recovery rate
        self.r0 = 1.0  # Base natural recovery rate
        self.base_emissions = 2.0  # Base CO2 emissions per tourist
        self.additional_emissions = 0.5  # Additional CO2 emissions per tourist

        # Social parameters
        self.a1 = -0.0001  # Resident satisfaction quadratic coefficient
        self.a2 = 0.01  # Resident satisfaction linear coefficient
        self.a3 = 0.1  # Tourist satisfaction spending coefficient
        self.a4 = -0.0005  # Tourist satisfaction crowding coefficient
        self.b1 = 90  # Resident satisfaction base level
        self.b2 = 80  # Tourist satisfaction base level

        # Infrastructure parameters
        self.k4 = 0.2  # Infrastructure development rate
        self.k5 = 0.3  # Infrastructure investment proportion
        self.k6 = 0.2  # Environmental protection investment proportion


# Define the complete system model
class TourismSystem:
    def __init__(self, constants):
        self.c = constants

    def calculate_revenue(self, Nt, tau_t, Pt):
        """Calculate total revenue including hidden costs"""
        # Revenue equation: Re(Nt,τt,Pt) = (τt + Pt)Nt - Pb - Pe - Ch
        Pb = self.c.k5 * (tau_t * Nt)  # Infrastructure investment
        Pe = self.c.k6 * (tau_t * Nt)  # Environmental protection investment
        Ch = self.c.c1 * Nt + self.c.c2 * Nt**2  # Hidden costs

        return (tau_t + Pt) * Nt - Pb - Pe - Ch

    def calculate_environmental_quality(self, Nt, Pe, E_prev, dt=1.0):
        """Calculate environmental quality change"""
        # Environmental quality equation: dE/dt = k₁Pe - k₂(CO₂p·Nt) + k₃Rnature
        CO2p = self.c.base_emissions + self.c.additional_emissions
        Rnature = self.c.r0 * (1 - Nt / self.c.NTMAX)

        dE = (self.c.k1 * Pe - self.c.k2 * (CO2p * Nt) + self.c.k3 * Rnature) * dt

        return E_prev + dE

    def calculate_satisfaction(self, Nt, Pt):
        """Calculate social satisfaction scores"""
        # Social satisfaction equations
        Sresidents = self.c.a1 * Nt**2 + self.c.a2 * Nt + self.c.b1
        Stourists = self.c.a3 * Pt + self.c.a4 * Nt + self.c.b2

        # Combined satisfaction with equal weights
        return (Sresidents + Stourists) / 2

    def calculate_infrastructure(self, Pb, Cinfra_prev, dt=1.0):
        """Calculate infrastructure capacity change"""
        # Infrastructure capacity equation: dCinfra/dt = k₄Pb
        return Cinfra_prev + self.c.k4 * Pb * dt


def evaluate(individual, system, prev_state):
    """Evaluate a solution according to all objectives"""
    Nt, tau_t, Pt = individual

    # Calculate system state
    Re = system.calculate_revenue(Nt, tau_t, Pt)
    E = system.calculate_environmental_quality(
        Nt, system.c.k6 * (tau_t * Nt), prev_state["E"]
    )
    S = system.calculate_satisfaction(Nt, Pt)
    Cinfra = system.calculate_infrastructure(
        system.c.k5 * (tau_t * Nt), prev_state["Cinfra"]
    )

    # Normalize objectives
    Re_norm = Re / system.c.REMAX
    E_norm = E / system.c.EMAX
    S_norm = S / system.c.SMAX
    Nt_norm = Nt / system.c.NTMAX

    return Re_norm, E_norm, S_norm, Nt_norm


def check_constraints(individual, system, prev_state):
    """Check if solution satisfies all constraints"""
    Nt, tau_t, Pt = individual

    # Calculate key metrics
    Re = system.calculate_revenue(Nt, tau_t, Pt)
    E = system.calculate_environmental_quality(
        Nt, system.c.k6 * (tau_t * Nt), prev_state["E"]
    )
    Sresidents = system.c.a1 * Nt**2 + system.c.a2 * Nt + system.c.b1
    CO2p = system.c.base_emissions + system.c.additional_emissions

    # Check all constraints
    constraints_satisfied = (
        Re >= 0,  # Financial constraint
        tau_t <= 50,  # Maximum tax constraint
        Pt <= 500,  # Maximum price constraint
        0 <= Nt <= system.c.NTMAX,  # Tourism capacity constraint
        Nt * CO2p <= system.c.CO2MAX,  # Environmental constraint
        E >= system.c.EMIN,  # Minimum environmental quality
        Sresidents >= 60,  # Minimum resident satisfaction
    )

    return all(constraints_satisfied)


# Setup optimization
def setup_optimization():
    """Setup the DEAP optimization framework"""
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    constants = ModelConstants()
    system = TourismSystem(constants)

    toolbox = base.Toolbox()

    # Register genetic operators
    toolbox.register("attr_float", np.random.uniform)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            lambda: toolbox.attr_float(0, constants.NTMAX),
            lambda: toolbox.attr_float(0, 50),
            lambda: toolbox.attr_float(100, 500),
        ),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Initial state
    prev_state = {
        "E": constants.EMAX * 0.8,  # Start at 80% environmental quality
        "Cinfra": constants.NTMAX * 0.7,  # Start at 70% infrastructure capacity
    }

    toolbox.register("evaluate", evaluate, system=system, prev_state=prev_state)
    toolbox.register("mate", tools.cxSimulatedBinaryNG)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[0, 0, 100],
        up=[constants.NTMAX, 50, 500],
        eta=20.0,
    )
    toolbox.register("select", tools.selNSGA2)

    return toolbox, system, prev_state


def main():
    """Main optimization routine"""
    toolbox, system, prev_state = setup_optimization()

    # Initialize population
    pop_size = 100
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = [toolbox.evaluate(ind) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    ngen = 50
    for gen in range(ngen):
        offspring = algorithms.varOr(
            pop, toolbox, lambda_=pop_size, cxpb=0.7, mutpb=0.3
        )

        # Evaluate offspring
        fits = [toolbox.evaluate(ind) for ind in offspring]
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        # Select next generation
        pop = toolbox.select(pop + offspring, k=pop_size)

        # Print progress
        if gen % 10 == 0:
            print(f"Generation {gen}")

    return pop, system


def analyze_results(final_pop, system):
    """Analyze and visualize optimization results"""
    # Extract Pareto front
    pareto_front = tools.sortNondominated(
        final_pop, len(final_pop), first_front_only=True
    )[0]

    print("\nPareto-optimal solutions:")
    for ind in pareto_front:
        Nt, tau_t, Pt = ind
        print(f"\nTourist number: {Nt:.0f}")
        print(f"Tourist tax: ${tau_t:.2f}")
        print(f"Average spending: ${Pt:.2f}")
        print(f"Objectives: {ind.fitness.values}")

        # Calculate additional metrics
        Re = system.calculate_revenue(Nt, tau_t, Pt)
        print(f"Total revenue: ${Re:.2f}")

    return pareto_front


if __name__ == "__main__":
    final_pop, system = main()
    pareto_front = analyze_results(final_pop, system)
