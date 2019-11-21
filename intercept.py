import numpy as np
from scipy.optimize import minimize

def f(params):
    return 0.5*params[0]**2

#params = {t, vxprime, vyprime, vzprime}

x0, y0, z0, vx, vy, vz, x0prime, y0prime, z0prime = 0, 0, 0, 100, 100, 100, 300, 300, 0

cons = ({'type': 'eq', 'fun': lambda x: x0prime - x0 + x[1]*x[0] - vx*x[0]},
        {'type': 'eq', 'fun': lambda x: y0prime - y0 + x[2]*x[0] - vy*x[0]},
        {'type': 'eq', 'fun': lambda x: z0prime - z0 + x[3]*x[0] - vz*x[0]},
        {'type': 'ineq', 'fun': lambda x: x[0]},
        # {'type': 'ineq', 'fun': lambda x: x[1]},
        # {'type': 'ineq', 'fun': lambda x: x[2]},
        # {'type': 'ineq', 'fun': lambda x: np.pi-x[1]},
        # {'type': 'ineq', 'fun': lambda x: 200-x[2]})
        {'type' : 'ineq', 'fun': lambda x: 200**2 - x[1]**2 - x[2]**2 - x[3]**2})

res = minimize(f, (1, 50, 50, 50), method='SLSQP', constraints=cons)

print(res)