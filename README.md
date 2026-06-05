# NumPy Study Guide

A structured breakdown of the original `numpy.py` file into topic-based modules.

---

## File Overview

| File | Topic | Key Concepts |
|------|-------|--------------|
| `01_array_fundamentals.py` | Array Basics | Creation, indexing, slicing, views vs copies, `ndim` / `shape` / `size` / `dtype` |
| `02_array_creation.py` | Array Constructors | `zeros`, `ones`, `empty`, `arange`, `sort`, `concatenate`, `reshape` |
| `03_array_operations.py` | Operations | Arithmetic, `sum/min/max`, broadcasting, `unique`, `transpose`, `flip`, `save/load` |
| `04_advanced_indexing.py` | Indexing & Masks | Boolean masking, fancy indexing, `argsort`, row/column filtering |
| `05_memory_and_strides.py` | Memory Layout | Strides, C/F-contiguous, `OWNDATA` flag, view vs copy |
| `06_multidimensional_reductions.py` | Axis Logic | Axis collapse, 3D arrays, factory/sensor challenges, standardization |
| `07_linear_algebra.py` | Linear Algebra | Dot product, `@` operator, element-wise vs matrix multiply, cache locality |
| `08_random_sampling_simulation.py` | Simulation | `default_rng`, normal distribution, city health risk capstone project |

---

## Recommended Study Order

```
01 → 02 → 03 → 04 → 06 → 05 → 07 → 08
```

Start with fundamentals (01–03), then master indexing (04), then axis logic (06), then internals (05), then math (07), and finish with the applied simulation (08).

---

## Quick Concept Reference

### Views vs Copies
```python
b = a[:]       # VIEW  — modifying b modifies a
b = a.copy()   # COPY  — independent
```

### Axis Logic (the "destroy" rule)
```python
# Shape (Factories=3, Days=5, Products=4)
arr.sum(axis=(0, 1))  # → shape (4,)  keep Products
arr.mean(axis=(1, 2)) # → shape (3,)  keep Factories
```

### Boolean Masking
```python
a[a > 5]                     # elements > 5
a[(a > 2) & (a < 10)]        # between 2 and 10
scores[scores[:, 0] > 80, 1] += 5  # update specific cells
```

### Broadcasting
```python
data = np.array([[1,2],[3,4],[5,6]])
data + np.array([[1,1]])  # row broadcasts across all rows
```

### Random (modern API)
```python
rng = np.random.default_rng(seed=42)
rng.integers(0, 10, size=5)
rng.normal(mean=45, std=15, size=1000)
```

---

## Capstone Project (File 08)

Simulates health risk scores for **100,000 people** across 5 cities using:
- Normal distribution for ages
- Fancy indexing to map city stress factors
- Boolean masking to identify the "danger zone" (risk > 1.2)
- Policy simulation: what if City 3 gets a green park?

---

## Prerequisites

```bash
pip install numpy scikit-learn
```

```python
import numpy as np
```
