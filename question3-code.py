# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# parameters
gamma = 2.0
beta = 0.985**20
r = 1.025**20 - 1.0

l = np.array([0.8027, 1.0, 1.2457])
NL = 3
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

pension = 0.498606

# utility function
def utility(c, gamma):
    return max(c, 1e-4)**(1.0 - gamma) / (1.0 - gamma)

# asset grid
a_l = 0.0
a_u = 2.0
NA = 300
a = np.linspace(a_l, a_u, NA)

# initialization
JJ = 3
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))

# period 3
for ia in range(NA):
    for il in range(NL):
        c3 = pension + (1 + r) * a[ia]
        v[2, ia, il] = utility(c3, gamma)
        aplus[2, ia, il] = 0.0

# period 2
for ia in range(NA):
    for il in range(NL):
        reward = np.zeros(NA)
        income = (1 - 0.3) * l[il]
        for iap in range(NA):
            cons = income + (1 + r) * a[ia] - a[iap]
            reward[iap] = utility(cons, gamma) + beta * v[2, iap, il]
        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# period 1
for ia in range(NA):
    for il in range(NL):
        reward = np.zeros(NA)
        for iap in range(NA):
            EV = sum(P[il, ilp] * v[1, iap, ilp] for ilp in range(NL))
            cons = l[il] + (1 + r) * a[ia] - a[iap]
            reward[iap] = utility(cons, gamma) + beta * EV
        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]

# figure
plt.figure()
plt.plot(a, aplus[0, :, 0], label='Low productivity', linewidth=2)
plt.plot(a, aplus[0, :, 1], label='Medium productivity', linewidth=2)
plt.plot(a, aplus[0, :, 2], label='High productivity', linewidth=2)
plt.title('Policy Function with Pension (Age 1)', fontsize=14)
plt.xlabel('Assets in Period 1 (a₁)', fontsize=12)
plt.ylabel('Assets in Period 2 (a₂)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()