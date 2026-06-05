# -*- coding: utf-8 -*-
"""
NumPy - Advanced Indexing & Boolean Masks
Covers: boolean masking, fancy indexing, argsort,
        conditional filtering, row/column selection
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Boolean Masking
# ──────────────────────────────────────────────

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(a[a < 5])                # Elements less than 5
print(a[a >= 5])               # Elements >= 5
print(a[a % 2 == 0])           # Divisible by 2
print(a[(a > 2) & (a < 10)])   # Between 2 and 10

# ──────────────────────────────────────────────
# 2. Fancy Indexing (Row/Column Selection)
# ──────────────────────────────────────────────

mat = np.array([[1, 2],
                [3, 4],
                [5, 6]])

# mat[[0,2]]  → fetch rows 0 and 2  (list of row indices)
# mat[0, 2]   → element at row 0, col 2  (scalar access)
print(mat[[0, 2]])

# Paired index access: (0,0) and (2,1)
mat = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

rows = [0, 2]
cols = [0, 1]
print(mat[rows, cols])    # → [10, 80]

rows = [1, 2]
cols = [2, 2]
print(mat[rows, cols])    # → [60, 90]

# ──────────────────────────────────────────────
# 3. Row Filtering with Column Conditions
# ──────────────────────────────────────────────

mat = np.array([[1, 2], [-3, 5], [-5, 2], [6, 7]])

# Find all rows where first element is negative
negative_rows = mat[mat[:, 0] < 0]
print(negative_rows)

# Extract last element of those rows
print(negative_rows[:, 1])

# ──────────────────────────────────────────────
# 4. Challenge: Student Scores
# ──────────────────────────────────────────────
# Rows = Students, Col 0 = Midterm, Col 1 = Final

scores = np.array([
    [85, 92],  # Student 0
    [42, 55],  # Student 1
    [78, 81],  # Student 2
    [30, 40],  # Student 3
    [90, 88]   # Student 4
])

# Students who failed the midterm (score < 50)
failed_midterm = scores[scores[:, 0] < 50]
print("Failed midterm:\n", failed_midterm)

# Give 5-point bonus on Final to students with Midterm > 80
scores[scores[:, 0] > 80, 1] += 5
print("After bonus:\n", scores)

# ──────────────────────────────────────────────
# 5. argsort — Sort by Index
# ──────────────────────────────────────────────
"""
np.argsort(arr) → indices that would sort the array.
Use those indices with fancy indexing to reorder another array.
"""

names  = np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eva'])
scores = np.array([55, 92, 40, 88, 71])

sort_index = np.argsort(scores)
print(names[sort_index])          # Lowest to highest
print(names[sort_index][::-1])    # Highest to lowest
