# Save this as moroccan_tsp.py
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List
import math

# Moroccan cities data
CITIES = [
    {"name": "Casablanca", "coords": [33.5731, -7.5898]},
    {"name": "Rabat", "coords": [34.0209, -6.8416]},
    {"name": "Fes", "coords": [34.0181, -5.0078]},
    {"name": "Marrakech", "coords": [31.6295, -7.9811]},
    {"name": "Tangier", "coords": [35.7595, -5.8340]},
    {"name": "Agadir", "coords": [30.4278, -9.5981]},
    {"name": "Meknes", "coords": [33.8935, -5.5547]},
    {"name": "Oujda", "coords": [34.6867, -1.9114]},
    {"name": "Kenitra", "coords": [34.2610, -6.5802]},
    {"name": "Tetouan", "coords": [35.5889, -5.3626]},
]

# GA Parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 200
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5

class City:
    def __init__(self, name: str, lat: float, lon: float):
        self.name = name
        self.lat = lat
        self.lon = lon

class Route:
    def __init__(self):
        self.path = []
        self.fitness = float('inf')
        
    def calculate_fitness(self) -> float:
        total_distance = 0
        for i in range(len(self.path)):
            city1 = self.path[i]
            city2 = self.path[(i + 1) % len(self.path)]
            total_distance += haversine_distance(city1, city2)
        self.fitness = total_distance
        return self.fitness

def haversine_distance(city1: City, city2: City) -> float:
    R = 6371
    lat1, lon1 = np.radians(city1.lat), np.radians(city1.lon)
    lat2, lon2 = np.radians(city2.lat), np.radians(city2.lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def create_initial_population(cities: List[City], pop_size: int) -> List[Route]:
    population = []
    for _ in range(pop_size):
        route = Route()
        route.path = random.sample(cities, len(cities))
        route.calculate_fitness()
        population.append(route)
    return population

def tournament_selection(population: List[Route]) -> Route:
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return min(tournament, key=lambda x: x.fitness)

def ordered_crossover(parent1: Route, parent2: Route) -> Route:
    size = len(parent1.path)
    child = Route()
    child.path = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child.path[start:end+1] = parent1.path[start:end+1]
    parent2_cities = [city for city in parent2.path if city not in child.path[start:end+1]]
    j = 0
    for i in range(size):
        if child.path[i] is None:
            child.path[i] = parent2_cities[j]
            j += 1
    child.calculate_fitness()
    return child

def swap_mutation(route: Route) -> None:
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route.path)), 2)
        route.path[i], route.path[j] = route.path[j], route.path[i]
        route.calculate_fitness()

def main():
    # Create cities
    cities = [City(city["name"], city["coords"][0], city["coords"][1]) for city in CITIES]
    
    # Initialize population
    population = create_initial_population(cities, POPULATION_SIZE)
    best_fitness_history = []
    
    # Main GA loop
    for generation in range(MAX_GENERATIONS):
        new_population = []
        best_route = min(population, key=lambda x: x.fitness)
        new_population.append(best_route)
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = ordered_crossover(parent1, parent2)
            swap_mutation(child)
            new_population.append(child)
        
        population = new_population
        best_fitness = min(route.fitness for route in population)
        best_fitness_history.append(best_fitness)
        
        if generation % 20 == 0:
            print(f"Generation {generation}: Best Distance = {best_fitness:.2f} km")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history)
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Distance (km)')
    plt.grid(True)
    plt.show()
    
    # Print final route
    best_route = min(population, key=lambda x: x.fitness)
    print("\nBest Route Found:")
    print(" -> ".join([city.name for city in best_route.path]))
    print(f"Total Distance: {best_route.fitness:.2f} km")

if __name__ == "__main__":
    main()
