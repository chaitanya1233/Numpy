# -*- coding: utf-8 -*-
"""
NumPy - Memory Layout, Strides & Array Flags
Covers: strides concept, C/F contiguous memory,
        OWNDATA flag, views vs copies
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Strides
# ──────────────────────────────────────────────
"""
Strides tell NumPy how many bytes to "jump" in memory
to reach the next element in each dimension.

  (12, 4) means:
    - 4 bytes  → move to next column (int32 = 4 bytes)
    - 12 bytes → move to next row (3 cols × 4 bytes = 12)
"""

matrix = np.array([[1, 2, 4], [5, 6, 7]], dtype=np.int32)

print("Shape:   ", matrix.shape)
print("Dtype:   ", matrix.dtype)
print("Strides: ", matrix.strides)

# ──────────────────────────────────────────────
# 2. Task: Shape-Shifter (Strides change, data doesn't)
# ──────────────────────────────────────────────

a = np.arange(24, dtype=np.int32)
print("1D strides:", a.strides)
# → (4,)  every element is 4 bytes apart

a_2d = a.reshape((4, 6))
print("2D strides:", a_2d.strides)
# → (24, 4): 24 bytes per row (6 × 4), 4 bytes per col

a_3d = a.reshape((2, 3, 4))
print("3D strides:", a_3d.strides)
# → (48, 16, 4)

# ──────────────────────────────────────────────
# 3. Array Flags
# ──────────────────────────────────────────────
"""
C_CONTIGUOUS → stored in C-style (Row-major)
OWNDATA      → True if the array owns its memory;
               False if it is a VIEW of another array
"""

mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
print("mat flags:\n", mat.flags)

mat2 = np.copy(mat)          # OWNDATA: True (copy owns data)
print("mat2 OWNDATA:", mat2.flags['OWNDATA'])

mat3 = mat                   # OWNDATA: True (same object)
print("mat3 OWNDATA:", mat3.flags['OWNDATA'])

# ──────────────────────────────────────────────
# 4. View vs Copy — OWNDATA Demo
# ──────────────────────────────────────────────

arr     = np.array([1, 2, 4, 55, 6, 7])
sub_arr = arr[0:2]  # This is a VIEW

print("arr OWNDATA:    ", arr.flags['OWNDATA'])    # True
print("sub_arr OWNDATA:", sub_arr.flags['OWNDATA']) # False
# sub_arr uses no extra memory → faster operations
