import numpy as np
import matplotlib.pyplot as plt

def kalman_update(x, P, F, B, u, Q, measurement, z, R, H):
    '''Performs one step of Kalman filter update (with measurements if provided).

    x : Current state estimate vector.
    P : Current state covariance matrix.
    F : State transition matrix.
    B : Control matrix.
    u : Control vector.
    Q : State uncertainty matrix.
    z : Current measured state vector.
    R : Measured state uncertainty matrix.
    H : Mapping matrix from states to measurements (not needed).'''

    assert x.shape == (6,1)
    assert P.shape == (6,6)
    assert F.shape == (6,6)

    xhat = np.matmul(F, x) + np.matmul(B, u)
    Phat = np.matmul(F, np.matmul(P, F.T)) + Q

    if measurement==0:
        return xhat, Phat

    else:
        Kprime = np.matmul(P, np.matmul(H.T, np.linalg.inv(np.matmul(H, np.matmul(P, H.T)) + R)))
        xhatprime = xhat + np.matmul(Kprime, z - np.matmul(H, xhat))
        Phatprime = Phat - np.matmul(Kprime, np.matmul(H, P))

        return xhatprime, Phatprime

def solve(measurements, measurements_uncertainty, measurements_idx, initx, initP, F, B, u, Q, N, H=np.eye(6)):
    '''Returns the state estimations given measurements and state dynamics.
    
    measurements : Matrix where each row is the state measurements. 0 if not available.
    measurements_uncertainty : Measurement uncertainties. 0 if not available.
    measurements_idx :  Length N array holding 1 if that time index has a measurement.
    initx : Mean for states estimate at time 0.
    initP : Covariance for states estimate at time 0.
    F : State transition matrix.
    B : Control matrix.
    u : Control vector.
    Q : State uncertainty matrix.
    N : Number of time indices.
    H : Mapping matrix from states to measurements (not needed).'''

    x_sol = [initx]
    P_sol = [initP]

    for k in range(N):
        x_now, P_now = kalman_update(x_sol[-1].reshape(-1,1), P_sol[-1], F, B, u, Q, \
                        measurements_idx[k], measurements[k].reshape(-1,1), measurements_uncertainty, H)

        x_sol.append(x_now.flatten())
        P_sol.append(P_now)

    return np.array(x_sol), np.array(P_sol)

def generate_data(xinit, vinit, delta, N, prop, noise):
    '''Generate true and noisy 3D data.

    xinit : 3D array of initial x,y,z positions.
    vinit : 3D array of initial x,y,z velocities.
    delta : Time delta.
    N : Number of time indices to consider.
    prop : Proportion of N, indicates how many noisy measurements to generate.
    noise : Variance of added noise.'''

    x = [xinit[0]]
    y = [xinit[1]]
    z = [xinit[2]]
    velx = [vinit[0]]
    vely = [vinit[1]]
    velz = [vinit[2]]
    idx = np.zeros(N)

    for k in range(N):
        x.append(x[-1] + velx[-1]*delta)
        y.append(y[-1] + vely[-1]*delta)
        z.append(z[-1] + velz[-1]*delta - 0.5*9.81*delta**2)
        velx.append(velx[-1])
        vely.append(vely[-1])
        velz.append(velz[-1] - 9.81*delta)
        if k % int(prop*N) == 0:
            idx[k] = 1

    actual = np.vstack((np.array(x),np.array(y),np.array(z),np.array(velx),np.array(vely),np.array(velz)))
    noisy = actual.T + noise*np.random.randn(N+1, 6)

    return idx, actual.T, noisy


xinit = np.array([0,0,0])
vinit = np.array([100,100,100])
delta = 0.1
N = 200
prop = 0.1
noise = 10
state_unc = 40

measurements_idx, actual, measurements = generate_data(xinit, vinit, delta, N, prop, noise)

F = np.array([[1,0,0,delta,0,0],
              [0,1,0,0,delta,0],
              [0,0,1,0,0,delta],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

B = np.array([0,0,0.5*delta**2,0,0,delta]).reshape(-1,1)
u = np.array([-9.81]).reshape(-1,1)

x, P = solve(measurements, noise*np.eye(6), measurements_idx, np.array([0,0,0,100,100,100]), \
             np.eye(6), F, B, u, state_unc*np.eye(6), N, H=np.eye(6))

#### PLOT RESULTS ####
plt.figure(figsize=(10,6))
#which axis to plot
i=0
j=2
plt.plot(x[:,i], x[:,j], c='b', linewidth=5, label='Kalman Estimate')
plt.plot(actual[:,i], actual[:,j], 'r', label='Actual', linewidth=5)
x_low = []
x_high = []
y_low = []
y_high = []
for k in range(N+1):
    samples = np.random.multivariate_normal(x[k], P[k], 1000)
    #lower = np.percentile(samples, 2.5, axis=0)
    #upper = np.percentile(samples, 97.5, axis=0)
    #x_low.append(lower[i])
    #x_high.append(upper[i])
    #y_low.append(lower[j])
    #y_high.append(upper[j])
    plt.scatter(samples[:, i], samples[:, j], s=5, alpha=0.01, c='b')

plt.scatter(measurements[1:][measurements_idx==1][:,i], measurements[1:][measurements_idx==1][:,j], c='k',\
            label='Noisy Measurements', s=150, zorder=5)
#plt.fill_between(actual[:,i], y_low, y_high, alpha=0.2, color='b', label='Kalman Estimate')
#plt.fill_betweenx(actual[:,j], x_low, x_high, alpha=0.2, color='b')
plt.legend()
plt.show()
