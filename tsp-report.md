# Solving the Traveling Salesman Problem for Moroccan Cities Using Genetic Algorithm

## Abstract
This project implements a genetic algorithm (GA) to solve the Traveling Salesman Problem (TSP) for major Moroccan cities. The implementation finds an optimal route connecting 10 major cities while minimizing the total travel distance.

## Problem Description
The TSP aims to find the shortest possible route that visits each city exactly once and returns to the starting city. For this implementation, we used actual coordinates of 10 major Moroccan cities including Casablanca, Rabat, Fes, and others.

## Implementation Details

### Data Structure
- Cities are represented as objects containing:
  - Name
  - Latitude
  - Longitude
- Routes are represented as ordered lists of cities

### Algorithm Components

1. **Population Initialization**
   - Random permutations of cities
   - Population size: 100 individuals

2. **Fitness Function**
   - Uses Haversine formula for accurate distance calculation
   - Accounts for Earth's curvature
   - Returns total route distance in kilometers

3. **Selection Method**
   - Tournament selection (size = 5)
   - Preserves best solution (elitism)

4. **Genetic Operators**
   - Ordered Crossover (OX) for generating offspring
   - Swap Mutation with 20% probability
   - Maintains route validity (no duplicate cities)

### Parameters
```python
POPULATION_SIZE = 100
MAX_GENERATIONS = 200
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5
```

## Results
The algorithm typically converges within 200 generations, producing:
- A complete route through all Moroccan cities
- Total distance optimization
- Visual representation via:
  - Interactive map with route overlay
  - Convergence graph showing fitness improvement

## Visualization
The implementation includes two visualization methods:
1. Interactive map showing the optimal route
2. Fitness progress graph displaying algorithm convergence

## Code Structure
```python
# Main components:
class City        # City representation
class Route       # Solution representation
def haversine_distance()    # Distance calculation
def ordered_crossover()     # Main genetic operator
def swap_mutation()         # Diversity maintenance
def tournament_selection()  # Parent selection
```

## Conclusion
The genetic algorithm successfully solves the TSP for Moroccan cities, providing:
- Optimized travel routes
- Visual route representation
- Real-time progress tracking
- Practical implementation for route planning

The solution demonstrates the effectiveness of genetic algorithms in solving complex optimization problems while maintaining practical applicability.
