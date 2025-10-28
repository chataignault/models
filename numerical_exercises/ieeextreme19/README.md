# IEEEXtreme 19 Programming Competition Solutions


<img src="img/xtreme_logo.gif" width="250px" />

Solutions to problems from the IEEEXtreme 19.0 programming competition. These represent completed submissions and post-competition work.

## Exercises

### Ladder (complete)
`ladder.py`, `ladder_solution.py`, `ladder.md`

Reverse-engineer and optimize a Monte Carlo simulation algorithm. 
The problem involves computing the probability 
that two random sequences of 1s and 2s, 
both summing to n, are identical. 
**Full mathematical derivation and solution provided in `ladder.md`.**

The exercise ranked second of the competition in terms of total points worth.

### Shailesh Triplet (complete)
`shailesh_triplet.py`

Find triplets (a, b, c) where a = b + c and k = a XOR b XOR c. 
Solution identifies valid triplets based on divisibility and power-of-two properties.

### Circular Permutation
`circular_permutation.py`

Arrange N friends around a circular table to minimize the maximum distance anyone needs to move. Tests both clockwise and counterclockwise arrangements starting from each position.

### Do You Know Expectation
`do_you_know_expectation.py`

Calculate the expected value of XOR operations raised to power k across all possible non-empty subsequences of an input array.

### Sequence Decomposition
`sequence_decomposition.py`

Decompose a sequence into fortune cookie patterns following a specific state machine (0→1→1→2→0→1→2). Uses greedy matching to identify and extract complete fortune patterns from the input stream.

### Stable Power Network
`stable_power_network.py`

Find the path in a weighted graph from node 1 to node N that minimizes maximum edge risk, with total time as tiebreaker. Modified Dijkstra's algorithm prioritizing (max_risk, total_time).

### Twin Occurrence
`twin_occurence.py`

Binary search to find the first and last occurrence indices of query values 
in a sorted array. 
Returns 1-indexed positions or -1 if not found.

