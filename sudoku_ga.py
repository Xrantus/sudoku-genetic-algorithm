import numpy as np
import random
import copy
import time

# --- HYPERPARAMETERS ---
POPULATION_SIZE = 1000
MAX_GENERATIONS = 10000
ELITISM_COUNT = 20
MUTATION_RATE = 0.5
RESTART_THRESHOLD = 600

class SudokuGA:
    def __init__(self, puzzle_str):
        self.original_grid = self.string_to_grid(puzzle_str)
        # Boolean mask: True for fixed (given) numbers, False for empty cells
        self.fixed_mask = (self.original_grid != 0) 
        self.population = []
        
        # Initialize the starting population
        self.initialize_population()

    def string_to_grid(self, puzzle_str):
        """Converts a Sudoku string into a 9x9 numpy array."""
        grid = np.zeros((9, 9), dtype=int)
        for i in range(81):
            row, col = i // 9, i % 9
            char = puzzle_str[i]
            if char.isdigit():
                grid[row, col] = int(char)
        return grid
    
    def get_row_missing_numbers(self, row_index):
        """Identifies missing numbers (1-9) in a specific row."""
        row_values = self.original_grid[row_index, :]
        existing = set(row_values[row_values != 0])
        all_nums = set(range(1, 10))
        return list(all_nums - existing)

    def initialize_population(self):
        """
        CHROMOSOME REPRESENTATION:
        Each individual is a 9x9 grid where every ROW satisfies the Sudoku constraint 
        (contains digits 1-9 unique). The algorithm optimizes columns and 3x3 blocks.
        """
        self.population = []
        for _ in range(POPULATION_SIZE):
            individual = self.original_grid.copy()
            for r in range(9):
                missing = self.get_row_missing_numbers(r)
                random.shuffle(missing)
                missing_idx = 0
                for c in range(9):
                    if not self.fixed_mask[r, c]:
                        individual[r, c] = missing[missing_idx]
                        missing_idx += 1
            self.population.append(individual)

    def calculate_fitness(self, individual):
        """
        FITNESS FUNCTION:
        Objective: Minimize conflicts to 0.
        Calculation: (Duplicate counts in Columns) + (Duplicate counts in 3x3 Blocks).
        """
        conflicts = 0
        # 1. Column Constraints
        for c in range(9):
            col = individual[:, c]
            conflicts += (9 - len(np.unique(col)))
            
        # 2. 3x3 Block Constraints
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = individual[r:r+3, c:c+3].flatten()
                conflicts += (9 - len(np.unique(block)))
                
        return conflicts

    def crossover(self, parent1, parent2):
        """
        Row-based Crossover. Preserves row consistency.
        """
        child = np.zeros((9, 9), dtype=int)
        crossover_point = random.randint(1, 8)
        child[:crossover_point, :] = parent1[:crossover_point, :]
        child[crossover_point:, :] = parent2[crossover_point:, :]
        return child

    def mutate(self, individual, current_mutation_rate):
        """
        Mutation Strategy: Swaps two numbers within a row.
        """
        if random.random() > current_mutation_rate:
            return individual

        row_idx = random.randint(0, 8)
        mutable_cols = [c for c in range(9) if not self.fixed_mask[row_idx, c]]
        
        if len(mutable_cols) >= 2:
            # Multi-Swap Strategy for stagnation
            if len(mutable_cols) >= 4 and random.random() < 0.3:
                c1, c2, c3, c4 = random.sample(mutable_cols, 4)
                individual[row_idx, c1], individual[row_idx, c2] = individual[row_idx, c2], individual[row_idx, c1]
                individual[row_idx, c3], individual[row_idx, c4] = individual[row_idx, c4], individual[row_idx, c3]
            else:
                # Standard Swap
                c1, c2 = random.sample(mutable_cols, 2)
                individual[row_idx, c1], individual[row_idx, c2] = individual[row_idx, c2], individual[row_idx, c1]
            
        return individual

    def solve(self):
        best_fitness_history = []
        stagnation_counter = 0
        best_global_score = float('inf')
        
        print(f"Starting Genetic Algorithm (Pop: {POPULATION_SIZE}, MaxGen: {MAX_GENERATIONS})...")

        for generation in range(MAX_GENERATIONS):
            # 1. Evaluate Fitness
            fitness_scores = [self.calculate_fitness(ind) for ind in self.population]
            min_fitness = min(fitness_scores)
            best_idx = fitness_scores.index(min_fitness)
            best_individual = self.population[best_idx]
            best_fitness_history.append(min_fitness)
            
            # Check for solution
            if min_fitness == 0:
                return True, generation, best_individual
            
            # Stagnation Check
            if min_fitness < best_global_score:
                best_global_score = min_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # --- STRATEGY: HARD RESTART ---
            if stagnation_counter >= RESTART_THRESHOLD:
                print(f"    [INFO] Stagnation Detected (Gen: {generation}). Restarting population...")
                self.initialize_population()
                stagnation_counter = 0
                best_global_score = float('inf') 
                continue

            # --- STRATEGY: DYNAMIC MUTATION RATE ---
            current_mutation_rate = MUTATION_RATE
            if stagnation_counter > 50:
                current_mutation_rate = min(0.9, MUTATION_RATE + (stagnation_counter / 1000))

            # 2. Create New Generation
            new_population = []
            
            # Elitism
            sorted_indices = np.argsort(fitness_scores)
            for i in range(ELITISM_COUNT):
                new_population.append(copy.deepcopy(self.population[sorted_indices[i]]))
            
            # Tournament Selection & Crossover
            while len(new_population) < POPULATION_SIZE:
                # Tournament 1
                candidates = random.sample(range(POPULATION_SIZE), 4)
                p1_idx = min(candidates, key=lambda x: fitness_scores[x])
                parent1 = self.population[p1_idx]

                # Tournament 2
                candidates = random.sample(range(POPULATION_SIZE), 4)
                p2_idx = min(candidates, key=lambda x: fitness_scores[x])
                parent2 = self.population[p2_idx]
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, current_mutation_rate)
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {min_fitness}")

        return False, MAX_GENERATIONS, self.population[np.argmin([self.calculate_fitness(i) for i in self.population])]

# --- HELPER & MAIN ---

def print_grid(grid):
    """Prints the Sudoku grid nicely."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            print(str(grid[i, j]) + " ", end="")
        print()

if __name__ == "__main__":
    # Example Sudoku
    # 0 represents empty cells
    sample_puzzle = "509030000000500003000100050900000241680000390012053768000005039098001524240000106"
    
    start_time = time.time()
    solver = SudokuGA(sample_puzzle)
    
    print("Initial Puzzle:")
    print_grid(solver.original_grid)
    print("\nSolving...")
    
    success, gen, solution = solver.solve()
    elapsed = time.time() - start_time
    
    print("\n" + "="*30)
    if success:
        print(f"SOLUTION FOUND in Generation {gen} ({elapsed:.2f}s)!")
        print("="*30)
        print_grid(solution)
    else:
        print(f"FAILED to find exact solution in {MAX_GENERATIONS} generations.")
        print("Best approximation:")
        print_grid(solution)