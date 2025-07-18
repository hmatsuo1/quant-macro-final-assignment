import numpy as np

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
mu = np.array([1.0 / NL] * NL)

# utility
def utility(c, gamma):
    return max(c, 1e-4)**(1.0 - gamma) / (1.0 - gamma)

# grid
a_l = 0.0
a_u = 2.0
NA = 300
a = np.linspace(a_l, a_u, NA)

# initialize
JJ = 3
v_nop = np.zeros((JJ, NA, NL))
v_wp = np.zeros((JJ, NA, NL))

# period 3
for ia in range(NA):
    for il in range(NL):
        v_nop[2, ia, il] = utility((1 + r) * a[ia], gamma)
        v_wp[2, ia, il] = utility(pension + (1 + r) * a[ia], gamma)

# period 2
for ia in range(NA):
    for il in range(NL):
        reward_nop = np.zeros(NA)
        reward_wp = np.zeros(NA)
        income_nop = l[il]
        income_wp = (1 - 0.3) * l[il]
        for iap in range(NA):
            cons_nop = income_nop + (1 + r) * a[ia] - a[iap]
            cons_wp = income_wp + (1 + r) * a[ia] - a[iap]
            reward_nop[iap] = utility(cons_nop, gamma) + beta * v_nop[2, iap, il]
            reward_wp[iap] = utility(cons_wp, gamma) + beta * v_wp[2, iap, il]
        v_nop[1, ia, il] = np.max(reward_nop)
        v_wp[1, ia, il] = np.max(reward_wp)

# period 1
for ia in range(NA):
    for il in range(NL):
        reward_nop = np.zeros(NA)
        reward_wp = np.zeros(NA)
        for iap in range(NA):
            EV_nop = np.sum(P[il, :] * v_nop[1, iap, :])
            EV_wp = np.sum(P[il, :] * v_wp[1, iap, :])
            cons = l[il] + (1 + r) * a[ia] - a[iap]
            reward_nop[iap] = utility(cons, gamma) + beta * EV_nop
            reward_wp[iap] = utility(cons, gamma) + beta * EV_wp
        v_nop[0, ia, il] = np.max(reward_nop)
        v_wp[0, ia, il] = np.max(reward_wp)

# initial asset = 0
init_index = 0
u_nop = v_nop[0, init_index, :]
u_wp = v_wp[0, init_index, :]

# weighted average
welfare_nop = np.sum(mu * u_nop)
welfare_wp = np.sum(mu * u_wp)

print(f"Welfare without pension: {welfare_nop:.6f}")
print(f"Welfare with pension:    {welfare_wp:.6f}")
