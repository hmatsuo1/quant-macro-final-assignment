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
value_function = np.zeros((num_periods, num_assets, num_types))
optimal_indices = np.zeros((num_periods, num_assets, num_types), dtype=int)
policy_function = np.zeros((num_periods, num_assets, num_types))

# period 3
for a_idx in range(num_assets):
    for type_idx in range(num_types):
        consumption = pension + (1 + interest_rate) * asset_grid[a_idx]
        value_function[2, a_idx, type_idx] = utility(consumption, gamma)
        policy_function[2, a_idx, type_idx] = 0.0

# period 2
for a_idx in range(num_assets):
    for type_idx in range(num_types):
        reward = np.zeros(num_assets)
        income = (1 - 0.3) * productivity[type_idx]
        for a_prime_idx in range(num_assets):
            consumption = income + (1 + interest_rate) * asset_grid[a_idx] - asset_grid[a_prime_idx]
            reward[a_prime_idx] = utility(consumption, gamma) + beta * value_function[2, a_prime_idx, type_idx]
        optimal_indices[1, a_idx, type_idx] = np.argmax(reward)
        policy_function[1, a_idx, type_idx] = asset_grid[optimal_indices[1, a_idx, type_idx]]
        value_function[1, a_idx, type_idx] = reward[optimal_indices[1, a_idx, type_idx]]

# period 1
for a_idx in range(num_assets):
    for type_idx in range(num_types):
        reward = np.zeros(num_assets)
        for a_prime_idx in range(num_assets):
            expected_value = sum(
                transition_matrix[type_idx, next_type] * value_function[1, a_prime_idx, next_type]
                for next_type in range(num_types)
            )
            consumption = productivity[type_idx] + (1 + interest_rate) * asset_grid[a_idx] - asset_grid[a_prime_idx]
            reward[a_prime_idx] = utility(consumption, gamma) + beta * expected_value
        optimal_indices[0, a_idx, type_idx] = np.argmax(reward)
        policy_function[0, a_idx, type_idx] = asset_grid[optimal_indices[0, a_idx, type_idx]]
        value_function[0, a_idx, type_idx] = reward[optimal_indices[0, a_idx, type_idx]]

# figure
plt.figure()
plt.plot(asset_grid, policy_function[0, :, 0], label='Low productivity', linewidth=2)
plt.plot(asset_grid, policy_function[0, :, 1], label='Medium productivity', linewidth=2)
plt.plot(asset_grid, policy_function[0, :, 2], label='High productivity', linewidth=2)
plt.title('Policy Function with Pension ', fontsize=14)
plt.xlabel('Assets in Period 1 (a₁)', fontsize=12)
plt.ylabel('Assets in Period 2 (a₂)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
