# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# parameters
gamma = 2.0
beta = 0.985**20
interest_rate = 1.025**20 - 1.0

productivity = np.array([0.8027, 1.0, 1.2457])
num_types = 3
transition_matrix = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

pension = 0.498606

# utility function
def utility(consumption, gamma):
    return max(consumption, 1e-4)**(1.0 - gamma) / (1.0 - gamma)

# asset grid
asset_min = 0.0
asset_max = 2.0
num_assets = 300
asset_grid = np.linspace(asset_min, asset_max, num_assets)

# initialization
num_periods = 3
V_no_pension = np.zeros((num_periods, num_assets, num_types))
V_with_pension = np.zeros((num_periods, num_assets, num_types))

# value function: period 3
for a_idx in range(num_assets):
    for type_idx in range(num_types):
        V_no_pension[2, a_idx, type_idx] = utility((1 + interest_rate) * asset_grid[a_idx], gamma)
        V_with_pension[2, a_idx, type_idx] = utility(pension + (1 + interest_rate) * asset_grid[a_idx], gamma)

# period 2
for a_idx in range(num_assets):
    for type_idx in range(num_types):
        reward_np = np.zeros(num_assets)
        reward_wp = np.zeros(num_assets)
        income_np = productivity[type_idx]
        income_wp = (1 - 0.3) * productivity[type_idx]
        for a_prime_idx in range(num_assets):
            # no pension
            cons_np = income_np + (1 + interest_rate) * asset_grid[a_idx] - asset_grid[a_prime_idx]
            reward_np[a_prime_idx] = utility(cons_np, gamma) + beta * V_no_pension[2, a_prime_idx, type_idx]
            # with pension
            cons_wp = income_wp + (1 + interest_rate) * asset_grid[a_idx] - asset_grid[a_prime_idx]
            reward_wp[a_prime_idx] = utility(cons_wp, gamma) + beta * V_with_pension[2, a_prime_idx, type_idx]
        V_no_pension[1, a_idx, type_idx] = np.max(reward_np)
        V_with_pension[1, a_idx, type_idx] = np.max(reward_wp)

# period 1
for a_idx in range(num_assets):
    for type_idx in range(num_types):
        reward_np = np.zeros(num_assets)
        reward_wp = np.zeros(num_assets)
        for a_prime_idx in range(num_assets):
            EV_np = sum(transition_matrix[type_idx, k] * V_no_pension[1, a_prime_idx, k] for k in range(num_types))
            EV_wp = sum(transition_matrix[type_idx, k] * V_with_pension[1, a_prime_idx, k] for k in range(num_types))
            cons = productivity[type_idx] + (1 + interest_rate) * asset_grid[a_idx] - asset_grid[a_prime_idx]
            reward_np[a_prime_idx] = utility(cons, gamma) + beta * EV_np
            reward_wp[a_prime_idx] = utility(cons, gamma) + beta * EV_wp
        V_no_pension[0, a_idx, type_idx] = np.max(reward_np)
        V_with_pension[0, a_idx, type_idx] = np.max(reward_wp)

# extract lifetime utility at initial asset = 0
initial_index = 0  # a=0

lifetime_util_no_pension = V_no_pension[0, initial_index, :]
lifetime_util_with_pension = V_with_pension[0, initial_index, :]

# weighted average
mu = np.array([1/3, 1/3, 1/3])
avg_util_no_pension = np.sum(mu * lifetime_util_no_pension)
avg_util_with_pension = np.sum(mu * lifetime_util_with_pension)

print(f"Average lifetime utility without pension: {avg_util_no_pension:.6f}")
print(f"Average lifetime utility with pension:    {avg_util_with_pension:.6f}")

# results
# Average lifetime utility without pension: -2.836918
# Average lifetime utility with pension:    -2.819299
