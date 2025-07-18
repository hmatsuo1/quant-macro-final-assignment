# import libraries
import numpy as np
import matplotlib.pyplot as plt

# parameters
gamma = 2.0
beta = 0.985**20
r = 1.025**20 - 1.0
l = np.array([0.8027, 1.0, 1.2457])  # labor productivity
NL = 3
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])
NA = 300
a_min = 0.0
a_max = 2.0
a = np.linspace(a_min, a_max, NA)
pension = 0.498606

# utility function
def utility(c, gamma):
    return max(c, 1e-4)**(1 - gamma) / (1 - gamma)

# function to solve model and return a₂ policy functions
def solve_policy(with_pension):
    v = np.zeros((3, NA, NL))
    ap = np.zeros((3, NA, NL))
    for ia in range(NA):
        for il in range(NL):
            c = (1 + r) * a[ia] + (pension if with_pension else 0.0)
            v[2, ia, il] = utility(c, gamma)
            ap[2, ia, il] = 0.0
    for ia in range(NA):
        for il in range(NL):
            income = (1 - 0.3) * l[il] if with_pension else l[il]
            reward = np.zeros(NA)
            for iap in range(NA):
                c = income + (1 + r) * a[ia] - a[iap]
                reward[iap] = utility(c, gamma) + beta * v[2, iap, il]
            idx = np.argmax(reward)
            ap[1, ia, il] = a[idx]
            v[1, ia, il] = reward[idx]
    for ia in range(NA):
        for il in range(NL):
            reward = np.zeros(NA)
            for iap in range(NA):
                EV = sum(P[il, ilp] * v[1, iap, ilp] for ilp in range(NL))
                c = l[il] + (1 + r) * a[ia] - a[iap]
                reward[iap] = utility(c, gamma) + beta * EV
            idx = np.argmax(reward)
            ap[0, ia, il] = a[idx]
            v[0, ia, il] = reward[idx]
    return ap[0, :, :]  # return only period 1 policy function a₂(a₁)

# solve for both cases
ap_young_wo = solve_policy(with_pension=False)  # without pension
ap_young_with = solve_policy(with_pension=True)  # with pension

# compute high - low gap for both cases
gap_wo = ap_young_wo[:, 2] - ap_young_wo[:, 0]
gap_with = ap_young_with[:, 2] - ap_young_with[:, 0]

# plot the results
plt.figure(figsize=(10, 5))
plt.plot(a, gap_wo, label='Without Pension', linewidth=2)
plt.plot(a, gap_with, label='With Pension', linewidth=2)
plt.title('Policy Gap (a₂^High − a₂^Low) by Initial Asset (a₁)', fontsize=14)
plt.xlabel('Initial Asset in Period 1 (a₁)', fontsize=12)
plt.ylabel('Gap in Next Asset (a₂)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
