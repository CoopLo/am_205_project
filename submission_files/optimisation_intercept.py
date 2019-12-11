import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def f(params):
    '''Objective to be minimised.

    params : Array of form [time, vxprime, vyprime, vzprime].'''
    return 0.5*params[0]**2

x0, y0, z0, vx, vy, vz, x0prime, y0prime, z0prime, g, delta = 0, 0, 0, 80, 80, 80, 1000, 100, 0, 9.81, 0.1

def kinematics(x_curr, y_curr, z_curr, vx, vy, vz, deltat, t_final):
    '''Integrate forward based on simple kinematics updates.'''

    t_list = np.arange(0, t_final, deltat)
    xs = [x_curr + vx*t for t in t_list]
    ys = [y_curr + vy*t for t in t_list]
    zs = [z_curr + vz*t - 0.5*g*t**2 for t in t_list]

    return xs, ys, zs

# Implement required constraints.
cons = ({'type': 'eq', 'fun': lambda x: x0prime - x0 + x[1]*(x[0] - delta) - vx*x[0]},
        {'type': 'eq', 'fun': lambda x: y0prime - y0 + x[2]*(x[0] - delta) - vy*x[0]},
        {'type': 'eq', 'fun': lambda x: z0prime - z0 + x[3]*(x[0] - delta) -
            vz*x[0] - 0.5*g*(x[0] - delta)**2 + 0.5*g*x[0]**2},
        {'type': 'ineq', 'fun': lambda x: x[0] - delta},
        {'type' : 'ineq', 'fun': lambda x: 100**2 - x[1]**2 - x[2]**2 - x[3]**2})

# Minimise objective subject to constraints.
res = minimize(f, (1, 50, 50, 50), method='SLSQP', constraints=cons)

print(res)

t, vxprime, vyprime, vzprime = res.x
deltat = 0.025

# Obtain enemy and response trajectories using forward integration.
xs,ys,zs = kinematics(x0, y0, z0, vx, vy, vz, deltat, t)
xprimes,yprimes,zprimes = kinematics(x0prime, y0prime, z0prime, vxprime,
        vyprime, vzprime, deltat, t-delta)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(xs, ys, zs, 'r', label='Enemy', linewidth=2)
ax.scatter(xs[0], ys[0], zs[0], c='r')
ax.text(xs[0]+10, ys[0]+10, zs[0]+10, 'Enemy Start')
ax.plot3D(xprimes, yprimes, zprimes, 'b', label='Response', linewidth=2)
ax.scatter(xprimes[0], yprimes[0], zprimes[0], c='b')
ax.text(xprimes[0]+10, yprimes[0]+10, zprimes[0]+10, 'Response Start')
ax.scatter(xs[-1], ys[-1], zs[-1], c='g')
ax.text(xs[-1]+10, ys[-1]+10, zs[-1]+10, 'Intercept')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
