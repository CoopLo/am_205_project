import numpy as np
from scipy.optimize import minimize

def f(params):
    return 0.5*params[0]**2

x0, y0, vx, vy, x0prime, y0prime = 0, 0, 100, 100, 500, 0

cons = ({'type': 'eq', 'fun': lambda x:  x0prime - x0 + x[2]*np.cos(x[1])*x[0] - vx*x[0]},
        {'type': 'eq', 'fun': lambda x:  y0prime - y0 + x[2]*np.sin(x[1])*x[0] - vy*x[0]},
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2]},
        {'type': 'ineq', 'fun': lambda x: np.pi-x[1]},
        {'type': 'ineq', 'fun': lambda x: 200-x[2]})

res = minimize(f, (1, np.pi/2,200), method='SLSQP', constraints=cons)

print(res)