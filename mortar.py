import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


def plot(steps, i, x, y, x_up, y_up):
    tqdm.write("PLOTTING Y: {}".format(y))
    leading_zeros = int(np.log(steps)/np.log(10))
    leading_zeros -= 0 if(i==0) else int(np.log(i)/np.log(10))
    zeros = '0'*leading_zeros
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='r')
    ax.set(xlim=(-0.5, x_up), ylim=(-0.5, y_up))
    ax.axhline(0, color='k')
    plt.savefig("./mortar/{}{}.png".format(zeros,i))
    plt.close(fig=fig)
    return True


def plot_both(steps, i, x, y, x_prime, y_prime, x_up, y_up):
    tqdm.write("PLOTTING Y: {}".format(y))
    leading_zeros = int(np.log(steps)/np.log(10))
    leading_zeros -= 0 if(i==0) else int(np.log(i)/np.log(10))
    zeros = '0'*leading_zeros
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='r')
    ax.scatter(x_prime, y_prime, color='b')
    ax.set(xlim=(-0.5, x_up), ylim=(-0.5, y_up))
    ax.axhline(0, color='k')
    plt.savefig("./mortar/{}{}.png".format(zeros,i))
    plt.close(fig=fig)
    return True


def kinematics(vx0, vy0, t, x0=0, y0=0, col_t=0):
    vx0 = np.array(vx0)
    #vy = vy0 - 9.81*t
    t = t*np.ones(vx0.size)
    t[1:] -= col_t
    x = x0 + vx0*t
    y = y0 + vy0*t - 1/2*9.81*t**2
    return np.array(x), np.array(y)
    

def mortar():
    steps = 500
    delta_t = 0.025

    # Enemy
    vx0 = 100
    vy0 = 100
    xs = [np.array(0)]
    ys = [np.array(0)]
    vxs = np.array(vx0)
    vys = np.array(vy0) 
    init_y = np.array(0)
    init_x = np.array(0)

    # Defense
    vx0_prime = 200*np.cos(2.61799388)
    vy0_prime = 200*np.sin(2.61799388)
    xs_prime = [np.array(500)]
    ys_prime = [np.array(0)]
    vxs_prime = np.array(vx0_prime)
    vys_prime = np.array(vy0_prime)
    init_y_prime = np.array(0)
    init_x_prime = np.array(500)

    boom = True
    collision_time = 0
    for i in tqdm(range(steps)):
        x, y = kinematics(vxs, vys, i*delta_t, init_x, init_y, collision_time*delta_t)
        x_list = x
        y_list = y

        x_prime, y_prime = kinematics(vxs_prime, vys_prime, i*delta_t,
                                     init_x_prime, init_y_prime, collision_time*delta_t)

        print("X_PRIME: {}".format(x_prime))
        x_list_prime = x_prime
        y_list_prime = y_prime

        try:
            mask = (y < 1e-2)
            init_y[mask] = 0
            y[mask] = np.zeros(len(mask))[mask]
            init_x[mask] = x[mask]
            #x[mask] = x_list[mask]
            print("Y: {}".format(y))
            print("init X: {}".format(init_x))
            print()
            for k in range(11):
                if(mask[k]):
                    vxs[k] = 0
                    vys[k] = 0
        except IndexError:
            pass

        if(boom and i>30):
            if(y_list < 1e-2):
                vy0 = 0
                init_x = [0]
                init_y = [0]
                for j in range(10):
                    init_x.append(x+5*(np.random.rand()-0.5))
                    init_y.append(y+2*np.random.rand())

                init_x = np.array(init_x)
                init_y = np.array(init_y)

                vxs = [0]
                vxs.extend(list((init_x[1:]-xs[i])))

                init_x = x*np.ones(11)
                init_y = y*np.ones(11)
                
                xs.append(np.array(init_x))
                ys.append(np.array(init_y))

                vys = [0]
                vys.extend(list(5*np.random.random(10)))
                boom = False
                collision_time = i
            else:
                xs.append(x)
                ys.append(y)
        else:
            xs.append(x)
            ys.append(y)

            xs_prime.append(x_prime)
            ys_prime.append(y_prime)
            
    end = True
    #print("MAX x: {}, MAX y: {}".format(np.max(xs), np.max(ys)))
    maxx = []
    maxy = []
    for i in range(steps):
        #print(xs[i])
        maxx.append(np.max(xs[i]))
        maxy.append(np.max(ys[i]))

    print(len(xs_prime))
    for i in tqdm(range(steps)):
        if(end):
            #end = plot(steps, i, xs[i], ys[i], 1.1*np.max(maxx), 1.1*np.max(maxy))
            end = plot_both(steps, i, xs[i], ys[i], xs_prime[i], ys_prime[i],
                            1.1*np.max(maxx), 1.1*np.max(maxy))
    os.system("cd ./mortar/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4")
        


if __name__ == '__main__':
    mortar()
