# -*- coding: utf-8 -*-
"""
NumPy - Random Sampling & Simulation
Covers: default_rng, integers, normal distribution,
        Capstone: City Health Risk Simulation
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. Random Number Generation (modern API)
# ──────────────────────────────────────────────
"""
Always use np.random.default_rng() — it's the modern,
reproducible way. Pass a seed for reproducibility.
"""

rng = np.random.default_rng(seed=42)

rfloat = rng.random()           # Single float [0, 1)
print(rfloat, type(rfloat))

# 3 random integers in [0, 10)
arr = rng.integers(low=0, high=10, size=3)
print(arr)

# 2D array of floats
arr_2d = rng.random(size=(252, 10000))
print("Shape:", arr_2d.shape)

# ──────────────────────────────────────────────
# 2. CAPSTONE PROJECT: City Health Risk Simulation
# ──────────────────────────────────────────────
"""
Scenario: Simulate health risk scores for 100,000 people
across 5 cities. Each city has a different stress factor.

- Higher score  → Higher risk
- Lower score   → Better health
- "Danger Zone" → score > 1.2
"""

rng = np.random.default_rng(seed=42)
n_people = 100_000

# Phase 1: Generate Population
city_ids     = rng.integers(0, 5, n_people)         # City: 0–4
ages         = rng.normal(45, 15, n_people)          # Mean=45, SD=15
base_health  = rng.random(n_people)                  # Risk: [0, 1)

# Phase 2: Map Stress Factors (Fancy Indexing)
stress_factors = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
mapped_stress  = stress_factors[city_ids]            # One line!

final_risk = base_health * mapped_stress

print(f"First 5 City IDs:          {city_ids[:5]}")
print(f"First 5 Stress Multipliers:{mapped_stress[:5]}")
print(f"First 5 Risk Scores:       {final_risk[:5]}")

# Phase 3: Identify Danger Zone (Boolean Masking)
danger_mask    = final_risk > 1.2
critical_count = np.sum(danger_mask)
percentage     = (critical_count / n_people) * 100

print(f"\nAlert: {critical_count} citizens ({percentage:.2f}%) are in the Danger Zone!")

# ──────────────────────────────────────────────
# 3. Policy Simulation: Green Park Initiative
# ──────────────────────────────────────────────
"""
What if City 3 (stress=1.5) gets a green park → stress drops to 1.0?
"""

new_stress = stress_factors.copy()
new_stress[new_stress == 1.5] = 1.0     # City 3 now relaxed

new_mapped = new_stress[city_ids]
new_risk   = new_mapped * base_health

danger_after = np.sum(new_risk > 1.2)
saved        = critical_count - danger_after

print(f"\nBefore parks: {critical_count} in danger")
print(f"After parks:  {danger_after} in danger")
print(f"Saved:        {saved} citizens!")

# ──────────────────────────────────────────────
# 4. Heatwave Stress Test (Broadcasting)
# ──────────────────────────────────────────────
"""
Heatwave multipliers per vital:
  Heart Rate:    × 1.10
  Blood Pressure:× 1.05
  Cholesterol:   × 1.00
"""

heatwave_factor = np.array([1.10, 1.05, 1.0])

# Simulate 5 cities × 3 vitals
vitals = np.random.rand(5, 3) * 100     # Shape (5, 3)
vitals_after = vitals * heatwave_factor  # Broadcasts automatically

print("\nVitals after heatwave:\n", np.round(vitals_after, 2))
