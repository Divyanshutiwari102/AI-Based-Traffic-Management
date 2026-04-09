from __future__ import annotations

import numpy as np

from config import (
    EPSILON,
    GA_BETA,
    GA_CYCLE_TIME,
    GA_GREEN_MAX,
    GA_GREEN_MIN,
    GA_MAX_ITER,
    GA_MUTATION_RATE,
    GA_POP_SIZE,
)

ROAD_CAPACITY_DEFAULT = 20
DEMAND_ADJUSTMENT_FACTOR = 0.9


def lane_pressure(counts: dict[str, int], capacity: int) -> float:
    weights = {"car": 1.0, "truck": 1.5, "bus": 2.0, "motorcycle": 0.5, "person": 0.3}
    weighted = sum(weights.get(cls_name, 1.0) * n for cls_name, n in counts.items())
    return min(weighted / max(capacity, 1), 1.0)


def _webster_lane_delay(green: float, cycle: float, demand: float, sat_flow: float = 0.5) -> float:
    if sat_flow <= 0 or green <= 0 or cycle <= 0:
        return float("inf")

    q = max(demand * DEMAND_ADJUSTMENT_FACTOR, EPSILON)
    g_over_c = green / cycle
    max_g_over_c = 0.99 * sat_flow / q
    if g_over_c >= max_g_over_c:
        g_over_c = max(max_g_over_c, 1e-6)

    denominator = 2 * (1 - (g_over_c * q / sat_flow))
    if denominator <= EPSILON:
        return float("inf")

    delay = (cycle * (1 - g_over_c) ** 2) / denominator
    return max(delay, 0.0)


def fitness(individual, density_vector, capacity):
    total_delay = 0.0
    cycle = float(np.sum(individual) + 4 * 3)

    for green, demand in zip(individual, density_vector):
        lane_delay = _webster_lane_delay(float(green), cycle, float(demand))
        if not np.isfinite(lane_delay):
            return -1e12
        total_delay += lane_delay * float(demand)

    return -float(total_delay)


def initialize_population(pop_size, num_lights, green_min, green_max, cycle_time, density_vector):
    population = []
    while len(population) < pop_size:
        green_times = np.random.randint(green_min, green_max + 1, num_lights)
        if np.sum(green_times) <= cycle_time:
            score = fitness(green_times, density_vector, ROAD_CAPACITY_DEFAULT)
            population.append((green_times, score))
    return sorted(population, key=lambda x: x[1], reverse=True)


def roulette_wheel_selection(population, scores, beta):
    shifted = np.array(scores) - np.max(scores)
    probs = np.exp(beta * shifted)
    probs /= np.sum(probs)
    return int(np.random.choice(len(population), p=probs))


def crossover(parent1, parent2, num_lights):
    point = np.random.randint(1, num_lights)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def mutate(individual, mutation_rate, green_min, green_max):
    num_lights = len(individual)
    mutated = individual.copy()
    mutations = max(1, int(mutation_rate * num_lights))
    for _ in range(mutations):
        idx = np.random.randint(0, num_lights)
        delta = np.random.choice([-1, 1]) * 0.05 * (green_max - green_min)
        mutated[idx] = np.clip(mutated[idx] + delta, green_min, green_max)
    return mutated


def genetic_algorithm(pop_size, num_lights, max_iter, green_min, green_max, cycle_time, mutation_rate, beta, density_vector):
    population = initialize_population(pop_size, num_lights, green_min, green_max, cycle_time, density_vector)
    best_sol = population[0]

    for _ in range(max_iter):
        scores = [ind[1] for ind in population]
        new_population = []

        while len(new_population) < pop_size:
            i1 = roulette_wheel_selection(population, scores, beta)
            i2 = roulette_wheel_selection(population, scores, beta)
            parent1, parent2 = population[i1][0], population[i2][0]
            child1, child2 = crossover(parent1, parent2, num_lights)

            for child in (child1, child2):
                child = mutate(child, mutation_rate, green_min, green_max)
                if np.sum(child) <= cycle_time:
                    score = fitness(child, density_vector, ROAD_CAPACITY_DEFAULT)
                    new_population.append((child, score))
                if len(new_population) >= pop_size:
                    break

        population += new_population
        population = sorted(population, key=lambda x: x[1], reverse=True)[:pop_size]
        if population[0][1] > best_sol[1]:
            best_sol = population[0]

    return best_sol


def optimize_traffic(densities):
    pop_size = GA_POP_SIZE
    num_lights = 4
    max_iter = GA_MAX_ITER
    green_min = GA_GREEN_MIN
    green_max = GA_GREEN_MAX
    cycle_time = GA_CYCLE_TIME
    mutation_rate = GA_MUTATION_RATE
    beta = GA_BETA

    best_sol = genetic_algorithm(
        pop_size,
        num_lights,
        max_iter,
        green_min,
        green_max,
        cycle_time,
        mutation_rate,
        beta,
        densities,
    )

    return {
        "north": int(best_sol[0][0]),
        "south": int(best_sol[0][1]),
        "west": int(best_sol[0][2]),
        "east": int(best_sol[0][3]),
    }
