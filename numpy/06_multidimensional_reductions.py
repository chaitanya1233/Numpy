# -*- coding: utf-8 -*-
"""
NumPy - Multidimensional Reductions & Axis Logic
Covers: axis collapse concept, 3D arrays,
        factory/city planner challenges, standardization
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. The "Sacrifice" Rule (Axis Logic)
# ──────────────────────────────────────────────
"""
When you specify axis=N in a reduction (sum, mean, etc.),
you are COLLAPSING (destroying) that dimension.

Shape: (2, 3, 4)
  - sum(axis=0) → shape (3, 4)   [Depth collapsed]
  - sum(axis=1) → shape (2, 4)   [Rows collapsed]
  - sum(axis=2) → shape (2, 3)   [Columns collapsed]

Index Removal Method (cheat sheet):
  Write the shape. Remove the index at the axis number.
"""

arr = np.arange(24).reshape(2, 3, 4)
print("Original shape:", arr.shape)

sum_axis0 = arr.sum(axis=0)
print("After axis=0:", sum_axis0.shape)  # (3, 4)

sum_axis2 = arr.sum(axis=2)
print("After axis=2:", sum_axis2.shape)  # (2, 3)

# ──────────────────────────────────────────────
# 2. Challenge A: Factory Production
# ──────────────────────────────────────────────
# Shape: (3 Factories, 5 Days, 4 Products)

arr = np.arange(60).reshape(3, 5, 4)

# Total production per product (keep Products → collapse Factories & Days)
total_production = np.sum(arr, axis=(0, 1))
print("Total per product:", total_production)  # shape (4,)

# Average production per factory (keep Factories → collapse Days & Products)
avg_production = np.mean(arr, axis=(1, 2))
print("Avg per factory:", avg_production)      # shape (3,)

# ──────────────────────────────────────────────
# 3. Challenge B: City Power Usage
# ──────────────────────────────────────────────
# Shape: (10 Cities, 20 Months, 30 Street_Light_IDs)

arr = np.random.rand(10, 20, 30)

# Total power usage per City → destroy (Month, Light_ID)
city_power = arr.sum(axis=(1, 2))
print("City power shape:", city_power.shape)    # (10,)

# Avg power per Month → destroy (City, Light_ID)
month_avg = arr.mean(axis=(0, 2))
print("Month avg shape:", month_avg.shape)      # (20,)

# ──────────────────────────────────────────────
# 4. Challenge: Feature Normalizer (Standardization)
# ──────────────────────────────────────────────
"""
Standardization formula per column:
    z = (x - mean) / std
Used in ML to prevent large-scale features from dominating.
"""

data = np.random.randint(20, 80, (100, 3))  # 100 people, 3 features

mean = np.mean(data, axis=0)   # Mean per feature (collapse rows)
std  = np.std(data, axis=0)    # Std per feature

standardized = (data - mean) / std

print("Standardized mean (should be ~0):", np.round(np.mean(standardized)))

# ──────────────────────────────────────────────
# 5. Challenge: Sensor Outlier Detector
# ──────────────────────────────────────────────
# Shape: (10 Sensors, 50 Time_Intervals, 5 Readings)

sensors = np.random.rand(10, 50, 5)

# Max reading per sensor (collapse Time & Readings)
max_per_sensor = np.max(sensors, axis=(1, 2))
print("Max per sensor shape:", max_per_sensor.shape)  # (10,)
