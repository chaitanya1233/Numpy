# -*- coding: utf-8 -*-
"""
NumPy - Array Fundamentals
Covers: Array creation, indexing, slicing, views vs copies,
        multidimensional arrays, array attributes
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Creating Arrays
# ──────────────────────────────────────────────

a = np.array([1, 2, 3, 4, 'x'])
print(a)
print(type(a))

# ──────────────────────────────────────────────
# 2. Indexing & Slicing (1D)
# ──────────────────────────────────────────────

a = np.array([1, 2, 3, 4, 5, 6])
print(a[0])       # First element
print(a[-1])      # Last element
print(a[:])       # All elements
print(a[::-1])    # Reversed
print(a[::-2])    # Reversed with step -2
print(a[::2])     # Every 2nd element

# Arrays are mutable
a[1] = 10
print(a)
print(a[:3])

# ──────────────────────────────────────────────
# 3. Views vs Copies
# ──────────────────────────────────────────────
"""
Slicing an array returns a VIEW (not a new list).
Modifying the view modifies the original array.
"""

b = a[:]

if id(a) == id(b):
    print("Same IDs")
else:
    print("Not same objects — but b is still a VIEW of a")

b[3] = 40
print(b)
print(a)  # a is also modified!

# ──────────────────────────────────────────────
# 4. Multidimensional Arrays
# ──────────────────────────────────────────────

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(type(a))
print(a.shape)   # (rows, cols)
print(a.size)    # Total elements

# Access element at row 2, col 3 → 12
print(a[2, 3])

# ──────────────────────────────────────────────
# 5. Array Attributes: ndim, shape, size, dtype
# ──────────────────────────────────────────────
"""
- Scalar  : 0-D array
- Vector  : 1-D array
- Matrix  : 2-D array
- Tensor  : N-D array (N > 2)
"""

a = np.array([1, 2, 3, 4, 5, 6])

print(a.ndim)             # Number of dimensions
print(a.shape)            # (6,) → 6 columns, 1 row
print(len(a.shape) == a.ndim)  # True
print(a.size)             # 6
print(a.dtype)            # Data type
