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
def kinematics(vx0, vy0, t, x0=0, y0=0, col_t=0, g=0, vt=0):
    # from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    vx0 = np.array(vx0)
    t = t*np.ones(vx0.size)
    t[1:] -= col_t
    x = x0 + vx0*t*np.exp(-g*t/vt)
    #try:
    #    vts = vt*np.ones(len(vy0))
    #except TypeError:
    #    vts = vt

    y = y0 + vt/g*(vy0 + vt)*(1 - np.exp(-g*t/vt)) - vt*t
    print("VY0: {}".format(vy0))
    print("VT: {}".format(vt))
    print("FACTOR: {}".format(1 - np.exp(-g*t/vt)))
    print("FIRST TERM: {}".format(vt/g*(vy0 + vt)))
    print("SECOND TERM: {}".format((1 - np.exp(-g*t/vt))))
    print("THIRD TERM: {}".format(-vt*t))

    print("t: {}".format(t))
    print("y: {}\n\n".format(y))
    return np.array(x), np.array(y)
    

def mortar():
    steps = 400
    delta_t = 0.01
    vx0 = 10
    vy0 = 10
    c = 0.5
    mass = np.array(50)
    g = 9.81
    vt = g*mass/c
    xs = [np.array(0)]
    ys = [np.array(0)]
    vxs = np.array(vx0)
    vys = np.array(vy0) 
    boom = True
    init_y = np.array(0)
    init_x = np.array(0)
    collision_time = 0
    breaks = 50
    j = 0
    for i in tqdm(range(steps)):
        x, y = kinematics(vxs, vys, i*delta_t, init_x, init_y, collision_time*delta_t, g, vt)
        x_list = x
        y_list = y
        if(not boom):
            j += 1

        try:
            if(j < 5):
                init_y[0] = y[0]
                init_x[0] = x[0]

            mask = (y < 1e-3)
            init_y[mask] = 0
            y[mask] = np.zeros(len(mask))[mask]
            init_x[mask] = x[mask]
            for k in range(breaks+1):
                if(mask[k]):
                    vxs[k] = 0
                    vys[k] = 0
        except IndexError:
            pass

        if(boom and i>30):
            if(y_list < 1e-3):
                vy0 = 0
                init_x = [x[0]]
                init_y = [y[0]]
                single_mass = mass
                mass = np.zeros(breaks+1)
                mass[0] = single_mass
                for j in range(breaks):
                    init_x.append(x)#+2*np.random.rand()-1)
                    init_y.append(y)#+0.1*np.random.rand())
                    mass[j+1] = mass[0]/breaks

                vt = mass*g/c
                init_x = np.array(init_x)
                init_y = np.array(init_y)

                vxs = [0]
                #vxs.extend(list((init_x[1:]-xs[i])))
                vxs.extend(list(5*np.random.randn(breaks)))

                #init_x = x*np.ones(11)
                #init_y = y*np.ones(11)
                
                xs.append(np.array(init_x))
                ys.append(np.array(init_y))

                vys = [0]
                vys.extend(list(5*np.random.random(breaks)+0.5))
                boom = False
                collision_time = i-1
            else:
                xs.append(x)
                ys.append(y)
        else:
            xs.append(x)
            ys.append(y)
            
    end = True
    #print("MAX x: {}, MAX y: {}".format(np.max(xs), np.max(ys)))
    maxx = []
    maxy = []
    for i in range(steps):
        maxx.append(np.max(xs[i]))
        maxy.append(np.max(ys[i]))

    for i in tqdm(range(steps)):
        if(end):
            end = plot(steps, i, xs[i], ys[i], 1.1*np.max(maxx), 1.1*np.max(maxy))
    os.system("cd ./mortar_drag/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4")


if __name__ == '__main__':
    mortar()
