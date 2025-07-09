# importing libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# parameters
gamma = 2.0
beta = 0.985**20
r = 1.025**20 - 1.0
l = np.array([0.8027, 1.0, 1.2457])  # productivity levels
NL = 3
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])  # transition prob

# utility function
def util(cons, gamma):
    return max(cons, 1e-4)**(1.0-gamma)/(1.0-gamma)

# asset grid
a_l = 0.0
a_u = 2.0
NA = 100
a = np.linspace(a_l, a_u, NA)

JJ = 3  # periods
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))

# backward induction
# period 3 (old age, consume all assets)
for ia in range(NA):
    for il in range(NL):
        v[2, ia, il] = util((1 + r) * a[ia], gamma)
        aplus[2, ia, il] = 0.0  # consume all

# period 2 (middle age)
for ia in range(NA):
    for il in range(NL):
        reward = np.zeros(NA)
        for iap in range(NA):
            reward[iap] = util(l[il] + (1 + r) * a[ia] - a[iap], gamma) + beta * v[2, iap, il]
        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# period 1 (young age)
for ia in range(NA):
    for il in range(NL):
        reward = np.zeros(NA)
        for iap in range(NA):
            EV = 0.0
            for ilp in range(NL):
                EV += P[il, ilp] * v[1, iap, ilp]
            reward[iap] = util(l[il] + (1 + r) * a[ia] - a[iap], gamma) + beta * EV
        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]

# figure
plt.figure()
plt.plot(a, aplus[0, :, 0], label='Low productivity', linewidth=2)
plt.plot(a, aplus[0, :, 1], label='Middle productivity', linewidth=2)
plt.plot(a, aplus[0, :, 2], label='High productivity', linewidth=2)
plt.title('Policy Function without Pension', fontsize=14)
plt.xlabel('Initial Asset (a₁)', fontsize=12)
plt.ylabel('Next Asset (a₂)', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()