# DAA Jackfruit

**Subject:** Design and Analysis of Algorithms
**Department:** Computer Science and Engineering
**University:** PES University, Electronic City Campus, Bengaluru
**Faculty:** Dr. Vandana M L

---

## About This Project

This project solves three algorithm problems as part of the DAA Jackfruit assessment. Each problem is implemented in C with both a brute-force approach and an optimised solution, along with complexity analysis, pseudocode, and test cases.

---

## Problems

### Q1 — Sliding Window X-Sum

**File:** `q1_sliding_window_xsum.c`

Given an array `nums` of n integers and two integers `k` (window size) and `x`:
For every contiguous subarray of size k, compute its **x-sum**:
- Count the frequency of each element in the window
- Select top-x most frequent elements (tie-break: prefer the larger value)
- x-sum = sum of all occurrences of those selected elements
- If distinct elements ≤ x, sum the entire window

**Approaches:**

| Approach | Time Complexity | Space |
|---|---|---|
| Brute Force | O(n · k log k) | O(k) |
| Sliding Window (optimised) | O(n · k log x) | O(k) |

**Key idea:** Instead of rebuilding the frequency table from scratch every window, the sliding window keeps it alive and makes exactly two updates per step — add the incoming element, remove the outgoing one.

**Data structures used:** `FreqEntry` array (value + count pairs), qsort with custom comparator, swap-and-shrink O(1) deletion.

---

### Q2 — Uber Routing System

**File:** `q2_uber_routing.c`

A city road network is modelled as a directed weighted graph. Nodes are intersections, edges are roads with travel times.

For each node, compute:
- `forward_latency[v]` = shortest travel time from node 1 to v
- `reverse_latency[v]` = shortest travel time from v to node r_nodes

A road `(u → v, weight w)` is **consistent** if it lies on at least one shortest path from node 1 to node r_nodes:

```
dist_fwd[u] + w + dist_rev[v] == total_shortest
```

**Approach:** Run Dijkstra twice — forward from node 1, and on the reversed graph from node r_nodes.

| Step | Algorithm | Time |
|---|---|---|
| Forward shortest paths | Dijkstra from node 1 | O((V+E) log V) |
| Reverse shortest paths | Dijkstra on flipped graph from r_nodes | O((V+E) log V) |
| Consistency check | Single pass over edges | O(E) |
| **Total** | | **O((V+E) log V)** |

**Data structures used:** Binary min-heap (built from scratch), adjacency list via linked arrays, lazy deletion for stale heap entries.

Verified against Floyd-Warshall brute force on all test cases.

---

### Q3 — Strassen's Matrix Multiplication

**File:** `q3_strassen.c`

Standard n×n matrix multiplication takes O(n³). Strassen's divide-and-conquer algorithm reduces this by splitting each matrix into four n/2 × n/2 quadrants and computing only **7 recursive multiplications** instead of 8.

**The 7 products** (for A = [[a,b],[c,d]], B = [[e,f],[g,h]]):

```
M1 = (a+d)(e+h)      M5 = (a+b)*h
M2 = (c+d)*e         M6 = (c-a)(e+f)
M3 = a*(f-h)         M7 = (b-d)(g+h)
M4 = d*(g-e)
```

**Result quadrants:**

```
C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6
```

**Recurrence:** T(n) = 7·T(n/2) + Θ(n²)
**Master Theorem:** log₂(7) ≈ 2.807 > 2 → Case 1 → **T(n) = Θ(n^2.807)**

| Algorithm | Complexity | Ops at n=1000 |
|---|---|---|
| Naive | O(n³) | ~1,000,000,000 |
| Strassen | O(n^2.807) | ~130,000,000 |

**Benchmark results:**

| n | Naive | Strassen | Speedup |
|---|---|---|---|
| 64 | 0.1 ms | 0.1 ms | ~1x |
| 128 | 0.9 ms | 1.0 ms | ~1x |
| 256 | 6.4 ms | 5.1 ms | 1.25x |
| 512 | 26.2 ms | 15.4 ms | 1.71x |

**Crossover point:** Naive is faster for n ≤ 64 due to recursion and allocation overhead. Threshold set to 64.

**Numerical stability:** Strassen does 18 matrix additions per recursion level across log₂(n) levels. Floating-point errors compound at each level, which is why BLAS/LAPACK still use naive multiply for precision-critical work.

---

## How to Compile and Run

**Mac / Linux:**
```bash
gcc -O2 -std=c99 q1_sliding_window_xsum.c -o q1 && ./q1
gcc -O2 -std=c99 q2_uber_routing.c        -o q2 && ./q2
gcc -O2 -std=c99 q3_strassen.c            -o q3 -lm && ./q3
```

**Windows (MinGW):**
```bash
gcc -O2 -std=c99 q1_sliding_window_xsum.c -o q1.exe && q1.exe
gcc -O2 -std=c99 q2_uber_routing.c        -o q2.exe && q2.exe
gcc -O2 -std=c99 q3_strassen.c            -o q3.exe -lm && q3.exe
```

All three files are self-contained — no external libraries needed (Q3 uses `-lm` for `math.h`).
