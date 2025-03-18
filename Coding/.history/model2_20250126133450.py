import random
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize


class GlobalTourismOptimizer:
    def __init__(self):
        # Original constants and coefficients
        self.Nmax = 8000000
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

        # Advanced optimization parameters
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.elite_size = 5
        self.tournament_size = 3

        # Adaptive penalty parameters
        self.initial_penalty = 1e4
        self.penalty_increase_rate = 2.0
        self.penalty_decrease_rate = 0.5
        self.min_penalty = 1e2
        self.max_penalty = 1e8

    def calculate_investments(self, Nt, tau_t, k5, k6, k7):
        """Calculate investment allocations"""
        Re = self.Pt * Nt
        P_waste = k5 * tau_t * Re
        P_water = k6 * tau_t * Re
        P_e = k7 * tau_t * Re
        return P_waste, P_water, P_e

    def evaluate_constraints(self, x):
        """Evaluate all constraints and return violations"""
        Nt, tau_t, k5, k6, k7 = x
        violations = []

        # Tourism constraints
        violations.append(max(0, Nt - self.Nmax))
        violations.append(max(0, 100000 - Nt))

        # Tax rate constraints
        violations.append(max(0, tau_t - 0.08))
        violations.append(max(0, 0.02 - tau_t))

        # Investment ratio constraints
        total_inv = k5 + k6 + k7
        violations.append(max(0, total_inv - 0.4))
        violations.append(max(0, 0.15 - total_inv))

        # Individual investment constraints
        for k in [k5, k6, k7]:
            violations.append(max(0, 0.05 - k))
            violations.append(max(0, k - 0.2))

        # Environmental constraints
        co2_emissions = Nt * self.CO2p
        violations.append(max(0, co2_emissions - self.CO2max))

        return np.array(violations)

    def objective_with_penalty(self, x, penalty_coefficient):
        """Calculate objective value with adaptive penalty"""
        Nt, tau_t, k5, k6, k7 = x

        # Calculate base objective
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

        # Calculate penalty
        violations = self.evaluate_constraints(x)
        penalty = penalty_coefficient * np.sum(violations**2)

        return -(Re + S - E) + penalty

    def create_individual(self):
        """Create a random solution within bounds"""
        return np.array(
            [
                random.uniform(100000, self.Nmax),
                random.uniform(0.02, 0.08),
                random.uniform(0.05, 0.2),
                random.uniform(0.05, 0.2),
                random.uniform(0.05, 0.2),
            ]
        )

    def tournament_selection(self, population, fitness_values):
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def mutate(self, individual):
        """Mutate an individual"""
        mutation_mask = np.random.random(5) < self.mutation_rate
        if np.any(mutation_mask):
            bounds = [
                (100000, self.Nmax),
                (0.02, 0.08),
                (0.05, 0.2),
                (0.05, 0.2),
                (0.05, 0.2),
            ]
            for i in range(5):
                if mutation_mask[i]:
                    individual[i] = random.uniform(bounds[i][0], bounds[i][1])
        return individual

    def hybrid_optimize(self):
        """Hybrid optimization combining GA with local search"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = float("inf")
        penalty_coefficient = self.initial_penalty

        for generation in range(self.generations):
            # Evaluate current population
            fitness_values = [
                self.objective_with_penalty(ind, penalty_coefficient)
                for ind in population
            ]

            # Update best solution
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < best_fitness:
                best_fitness = fitness_values[min_fitness_idx]
                best_solution = population[min_fitness_idx].copy()

            # Create new population
            new_population = []

            # Elitism
            elite_indices = np.argsort(fitness_values)[: self.elite_size]
            new_population.extend([population[i].copy() for i in elite_indices])

            # Generate rest of new population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            # Local search on best solution
            if generation % 10 == 0:
                result = minimize(
                    lambda x: self.objective_with_penalty(x, penalty_coefficient),
                    best_solution,
                    method="SLSQP",
                    bounds=[
                        (100000, self.Nmax),
                        (0.02, 0.08),
                        (0.05, 0.2),
                        (0.05, 0.2),
                        (0.05, 0.2),
                    ],
                    options={"maxiter": 100, "ftol": 1e-8},
                )
                if result.success and result.fun < best_fitness:
                    best_fitness = result.fun
                    best_solution = result.x

            # Update penalty coefficient
            violations = self.evaluate_constraints(best_solution)
            if np.any(violations > 1e-6):
                penalty_coefficient = min(
                    self.max_penalty, penalty_coefficient * self.penalty_increase_rate
                )
            else:
                penalty_coefficient = max(
                    self.min_penalty, penalty_coefficient * self.penalty_decrease_rate
                )

            population = new_population

            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {best_fitness:.2f}")

        # Final local optimization
        final_result = minimize(
            lambda x: self.objective_with_penalty(x, penalty_coefficient),
            best_solution,
            method="SLSQP",
            bounds=[
                (100000, self.Nmax),
                (0.02, 0.08),
                (0.05, 0.2),
                (0.05, 0.2),
                (0.05, 0.2),
            ],
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if final_result.success and final_result.fun < best_fitness:
            return final_result

        return type(
            "OptimizeResult",
            (),
            {
                "x": best_solution,
                "fun": best_fitness,
                "success": True,
                "message": "Optimization completed successfully",
            },
        )

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

        # Calculate and print key metrics
        Re = self.Pt * Nt
        P_waste, P_water, P_e = self.calculate_investments(Nt, tau_t, k5, k6, k7)

        print("\nKey Metrics:")
        print(f"Total Revenue: ${Re:,.2f}")
        print(f"CO2 Emissions: {Nt * self.CO2p:.2f}")
        print(
            f"Resident Satisfaction: {(self.a1 * Nt**2 + self.a2 * Nt + self.b1):.2f}"
        )

        # Check constraints
        violations = self.evaluate_constraints(result.x)
        if np.any(violations > 1e-6):
            print("\nWarning: Some constraints are violated:")
            print(f"Constraint violations: {violations}")

        print(f"\nObjective value: {-result.fun:.2f}")
        print(f"Optimization success: {result.success}")


def main():
    optimizer = GlobalTourismOptimizer()
    result = optimizer.hybrid_optimize()
    optimizer.analyze_solution(result)


if __name__ == "__main__":
    main()
