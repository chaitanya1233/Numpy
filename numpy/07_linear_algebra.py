# -*- coding: utf-8 -*-
"""
NumPy - Linear Algebra Basics
Covers: dot product, matrix multiplication (@ operator),
        element-wise vs matrix ops, cache locality note
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Operators
# ──────────────────────────────────────────────
"""
*  → Element-wise multiplication
@  → Matrix multiplication (same as np.matmul / np.dot)
"""

# ──────────────────────────────────────────────
# 2. Dot Product (manual way)
# ──────────────────────────────────────────────

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Without @ or np.dot:
dot_product = sum(v1 * v2)
print("Dot product:", dot_product)  # 32

# With np.dot:
print("np.dot:", np.dot(v1, v2))

# ──────────────────────────────────────────────
# 3. Matrix Multiplication
# ──────────────────────────────────────────────

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A @ B        # Modern NumPy way
print("A @ B:\n", C)

# ──────────────────────────────────────────────
# 4. Performance Note: Cache Locality
# ──────────────────────────────────────────────
"""
Matrix multiplication relies heavily on CPU cache.

- Contiguous access (C-style row-major):
  The CPU fetches a row; next values are already in cache.
  → FAST (cache hit)

- Strided access (transposed array):
  Elements of a "row" are far apart in memory.
  The CPU must fetch from RAM each time.
  → SLOW (cache miss)

Understanding this separates standard coders from
high-performance numerical programmers.
"""

import time

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

start = time.time()
_ = A @ B
print(f"Regular matmul: {time.time() - start:.4f}s")

start = time.time()
_ = A @ B.T.copy()  # Force contiguous before multiply
print(f"With contiguous transpose: {time.time() - start:.4f}s")
