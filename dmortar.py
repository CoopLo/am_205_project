import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt


#@jit
def plot(steps, i, x, y, x_up, y_up):
    #tqdm.write("PLOTTING Y: {}".format(y))
    leading_zeros = int(np.log(steps)/np.log(10))
    leading_zeros -= 0 if(i==0) else int(np.log(i)/np.log(10))
    zeros = '0'*leading_zeros
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='r')
    ax.set(xlim=(-0.5, x_up), ylim=(-0.5, y_up))
    ax.axhline(0, color='k')
    plt.savefig("./mortar_drag/{}{}.png".format(zeros,i))
    plt.close(fig=fig)
    return True


#@jit
def kinematics(vx0, vy0, t, x0=0, y0=0, g=0, vt=0):
    # from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    x = vx0*vt/g*(1 - np.exp(-g*t/vt))
    y = y0 + vt/g*(vy0 + vt)*(1 - np.exp(-g*t/vt)) - vt*t
    vx = vx0*np.exp(-g*t/vt)
    vy = vy0*np.exp(-g*t/vt) - vt*(1- np.exp(-g*t/vt))
    return x, y, vx, vy
    

def mortar():
    steps = 400
    delta_t = 0.025
    vx0 = 1000
    vy0 = 20
    c = 0.5
    mass = np.array(1)
    g = 9.81
    vt = g*mass/c
    xs = [0]
    ys = [0]
    vxs = [vx0]
    vys = [vy0]
    init_y = 0
    init_x = 0
    j = 0
    for i in tqdm(range(steps)):
        x, y, vx, vy = kinematics(vx0, vy0, i*delta_t, init_x, init_y, g, vt)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        if(y < 1e-2 and i>10):
            break

    end = True
    #print("MAX x: {}, MAX y: {}".format(np.max(xs), np.max(ys)))
    maxx = []
    maxy = []
    fig, ax = plt.subplots()
    ax.plot(vxs)
    ax.plot(vys)

    for i in range(len(xs)):
        maxx.append(np.max(xs[i]))
        maxy.append(np.max(ys[i]))

    for i in tqdm(range(len(xs))):
        if(end):
            end = plot(steps, i, xs[i], ys[i], 1.1*np.max(maxx), 1.1*np.max(maxy))
    os.system("cd ./mortar_drag/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4")


if __name__ == '__main__':
    mortar()
