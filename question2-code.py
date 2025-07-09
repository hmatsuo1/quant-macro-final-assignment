# importing libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# parameters
l = np.array([0.8027, 1.0, 1.2457])  # productivity levels
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])  # transition prob
r = 1.025**20 - 1
mu = np.array([1/3, 1/3, 1/3])  # ration of young, middle-aged, and old individuals

# 1. calculate tax
tax = 0.0
for i in range(3):  # 若年期のタイプ
    for j in range(3):  # 中年期のタイプ
        tax += mu[i] * P[i, j] * 0.3 * l[j]

# 2. calculate pension
pension = (1 + r) * tax

print(f"総税収（1人あたり）: {tax:.6f}")
print(f"一人あたり年金額: {pension:.6f}")

