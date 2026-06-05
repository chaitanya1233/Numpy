# -*- coding: utf-8 -*-
"""
NumPy - Array Operations
Covers: arithmetic, sum/min/max, broadcasting,
        unique values, transpose, reverse, save/load
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Arithmetic Operations
# ──────────────────────────────────────────────

data = np.array([1, 2])
ones = np.ones(2)

print(data + ones)
print(data - ones)
print(ones - data)
print(data * data)
print(data / ones)
print(data // ones)

# ──────────────────────────────────────────────
# 2. Aggregations: sum, min, max
# ──────────────────────────────────────────────

a = np.array([1, 2, 4, 5, 6])
print(np.sum(a))
print(a.sum())

# 2D: axis=0 → row-wise sum, axis=1 → col-wise sum
a = np.array([[1, 2, 4], [5, 6, 7]])
print(a.sum(axis=0))
print(a.sum(axis=1))

a = np.array([[1, 2], [5, 6], [7, 3]])
print(a.max(axis=0))  # Max per column
print(a.max(axis=1))  # Max per row

# ──────────────────────────────────────────────
# 3. Broadcasting
# ──────────────────────────────────────────────
"""
Broadcasting allows NumPy to operate on arrays of
different shapes, as long as dimensions are compatible
(equal OR one of them is 1).
"""

data = np.array([2, 6, 8])
print(data * 0.2)

# Matrix + Row vector (auto-broadcast)
data     = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
print(data + ones_row)

# ──────────────────────────────────────────────
# 4. Unique Values
# ──────────────────────────────────────────────

a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

print(np.unique(a))

unique_values, unique_index = np.unique(a, return_index=True)
print(f"{unique_values}: {unique_index}")

unique_values, occurrence_counts = np.unique(a, return_counts=True)
print(unique_values)
print(occurrence_counts)

# 2D unique
a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
print(np.unique(a_2d))                         # Flattened unique
print(np.unique(a_2d, axis=0))                 # Unique rows
print(np.unique(a_2d, axis=1))                 # Unique columns

# ──────────────────────────────────────────────
# 5. Transpose
# ──────────────────────────────────────────────

arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.reshape((2, 3)))
print(arr.reshape((3, 2)))

arr2 = np.arange(6).reshape((3, 2))
print(arr2.transpose())
print(arr2.T)

# ──────────────────────────────────────────────
# 6. Reversing Arrays (np.flip)
# ──────────────────────────────────────────────

arr = np.arange(5)
print(np.flip(arr))

arr_2d = np.arange(8).reshape((4, 2))
print(np.flip(arr_2d))               # Flip everything
print(np.flip(arr_2d, axis=0))       # Flip rows only
print(np.flip(arr_2d, axis=1))       # Flip columns only

# ──────────────────────────────────────────────
# 7. Save & Load
# ──────────────────────────────────────────────

a = np.array([1, 2, 3, 4, 5, 6])

np.save('data_A', a)               # Saves as data_A.npy
b = np.load('data_A.npy')
print(b)

csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.savetxt('new_csv_file.csv', csv_arr)

read_csv = np.loadtxt('new_csv_file.csv')
print(read_csv)
