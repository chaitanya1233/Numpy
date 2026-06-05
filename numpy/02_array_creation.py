# -*- coding: utf-8 -*-
"""
NumPy - Array Creation Methods
Covers: zeros, ones, empty, arange, linspace,
        sort, concatenate, reshape
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Basic Constructors
# ──────────────────────────────────────────────

a_zeros = np.zeros(2)
print(a_zeros)

a_ones = np.ones(2)
print(a_ones)

# Note: empty() is faster than zeros() — values are uninitialized
a_empty = np.empty(2)
print(a_empty)

# Range of elements
a = np.arange(4)
print(a)

# arange(start, stop, step)
a = np.arange(1, 3, 1)
print(a)

# Specify dtype
a = np.ones(2, dtype=np.int64)
print(a)

# ──────────────────────────────────────────────
# 2. Sorting
# ──────────────────────────────────────────────

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(np.sort(arr))  # Returns sorted copy (ascending)

# ──────────────────────────────────────────────
# 3. Concatenation
# ──────────────────────────────────────────────

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

result = np.concatenate((x, y))
print(result)

# ──────────────────────────────────────────────
# 4. Reshaping
# ──────────────────────────────────────────────
"""
IMPORTANT:
  - reshape() gives a new shape WITHOUT changing data.
  - The new shape must have the same total number of elements.

order parameter:
  - 'C' (default): Row-major (C-style)
  - 'F': Column-major (Fortran-style)
"""

a = np.arange(6)
print(a)
print(a.reshape(3, 2))

# ──────────────────────────────────────────────
# 5. 2D Random Matrices
# ──────────────────────────────────────────────

ones      = np.ones((3, 2))
zeros     = np.zeros((3, 2))
rand_mat  = np.random.default_rng().random((3, 2))

print(ones)
print(zeros)
print(rand_mat)
