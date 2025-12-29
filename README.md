# Hybrid Genetic Algorithm for Solving Sudoku

A Genetic Algorithm (GA) that solves 9×9 Sudoku puzzles, built for the *Evolutionary Computation* (CENG 482) course at İzmir Institute of Technology, Fall 2025.

Sudoku is an NP-complete constraint satisfaction problem. Standard GAs tend to stall in local optima where only a few cells conflict. This project tackles that with a constraint-aware encoding, adaptive operators, and a restart strategy.

## Method

- **Row-based permutation encoding:** every row is filled with a permutation of its missing digits, so row constraints always hold and only column and 3×3 block constraints remain to be optimized.
- **Fixed-cell mask:** the given clues are never modified by any operator.
- **Fitness:** number of missing digits across all columns and 3×3 blocks (0 = solved).
- **Selection:** tournament selection (size 4) with elitism (top 20 kept each generation).
- **Crossover:** row-wise single-point crossover that keeps rows intact.
- **Mutation:** swap of two non-fixed cells within a row, plus a multi-swap operator; the mutation rate increases adaptively while the population stagnates.
- **Hard restart:** if the best fitness does not improve for `RESTART_THRESHOLD` generations, the population is re-initialized to escape deep local optima.

## Results

Evaluated on subsets of the *4 Million Sudoku Puzzles* dataset with clue counts from 20 to 80. Alternative strategies (hyper-mutation, island model, memetic algorithm) were also compared.

- Puzzles with 50+ clues are typically solved in the initial population.
- Difficulty rises sharply below ~50 clues.
- The hard-restart strategy gave the best results, solving about **80%** of hard puzzles (<25 clues).

## Usage

```bash
pip install -r requirements.txt
python sudoku_ga.py
```

The script solves the sample puzzle defined at the bottom of `sudoku_ga.py`. To solve another puzzle, replace `sample_puzzle` with an 81-character string where `0` marks an empty cell.

Main hyperparameters (top of the file): `POPULATION_SIZE = 1000`, `MAX_GENERATIONS = 10000`, `ELITISM_COUNT = 20`, `MUTATION_RATE = 0.5`, `RESTART_THRESHOLD = 600`.

## Tech

Python, NumPy
