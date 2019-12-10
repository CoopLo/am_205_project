import numpy as np
from numba import jit
from enemy import rocket as enemy_rocket
from response import response as response_rocket
from multiprocessing import Process
from matplotlib import pyplot as plt
import os

def plot2():
    # Load enemy data
    ex = np.loadtxt("./parallel_test/enemy_x.txt")
    ey = np.loadtxt("./parallel_test/enemy_y.txt")
    ez = np.loadtxt("./parallel_test/enemy_z.txt")

    # Plot response path
    print("PLOTTING RESPONSE")
    r_x = np.loadtxt("./parallel_test/response_x.txt")
    r_y = np.loadtxt("./parallel_test/response_y.txt")
    r_z = np.loadtxt("./parallel_test/response_z.txt")

    i = 0
    px = np.loadtxt("./parallel_test/proj_x_{}.txt".format(i))
    py = np.loadtxt("./parallel_test/proj_y_{}.txt".format(i))
    pz = np.loadtxt("./parallel_test/proj_z_{}.txt".format(i))

    first_detection = int(np.loadtxt("./parallel_test/first_detection.txt"))

    print(len(ex))
    print(len(r_x))
    print(first_detection)
    for i in range(len(ex)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #if i < len(px):
            #ax.scatter(px[i], py[i], pz[i])  # , label="ENEMY ROCKET PROJECTION {}".format(i))

        # Plot detection hemisphere
        print("PLOTTING DOME")
        radius = 1500
        origin = [2000, 2000]
        phi, theta = np.mgrid[0.0:0.5 * np.pi:180j, 0.0:2.0 * np.pi:720j]
        dome_x = radius * np.sin(phi) * np.cos(theta) + origin[0]
        dome_y = radius * np.sin(phi) * np.sin(theta) + origin[1]
        dome_z = radius * np.cos(phi)
        ax.plot_surface(dome_x, dome_y, dome_z, alpha=0.15, color='g')

        ground_theta = np.linspace(0, 2 * np.pi, 100)
        ground_x = radius * np.cos(ground_theta) + origin[0]
        ground_y = radius * np.sin(ground_theta) + origin[1]
        ax.plot(ground_x, ground_y, alpha=0.5, color='g')
        # ax.set(zlim=(0, max(max(ez), max(pz))))
        ax.set(zlim=(0, max(ez)))

        ax.scatter(ex[i], ey[i], ez[i], label="ENEMY ROCKET PATH")
        ax.plot(ex[:i], ey[:i], ez[:i], '--', label="ENEMY ROCKET PATH")
        if i > first_detection:
            j = i - first_detection
            ax.scatter(r_x[j], r_y[j], r_z[j], color='r', label="RESPONSE ROCKET")
            ax.plot(r_x[:j], r_y[:j], r_z[:j], '--', color='r', label="RESPONSE ROCKET")
        ax.set_zlim(top=75)
        ax.set_xlim(right=2500)
        ax.set_ylim(top=2500)
        plt.savefig(f'./parallel_test/fig{i}')
        plt.close(fig=fig)



def plot_enemy_projection():
    # Load enemy data
    ex = np.loadtxt("./parallel_test/enemy_x.txt")
    ey = np.loadtxt("./parallel_test/enemy_y.txt")
    ez = np.loadtxt("./parallel_test/enemy_z.txt")

    """for i, j in enumerate(zip(ex, ey, ez)):
        print(i, j)"""


    fig = plt.figure()
    print("PLOTTING ENEMY")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ex, ey, ez, '--', color='r', alpha=0.5, label="ENEMY ROCKET PATH")

    # Get number of projections to plot
    dir_files = os.listdir("./parallel_test")
    projections = [s for s in dir_files if 'proj_x_' in s.lower()]
    max_proj = max([int(s.split('_')[2].split('.')[0]) for s in projections])

    # Plot projections
    for i in range(max_proj+1):
        px = np.loadtxt("./parallel_test/proj_x_{}.txt".format(i))
        py = np.loadtxt("./parallel_test/proj_y_{}.txt".format(i))
        pz = np.loadtxt("./parallel_test/proj_z_{}.txt".format(i))
        ax.plot(px, py, pz, alpha=0.25) #, label="ENEMY ROCKET PROJECTION {}".format(i))"""

    """# Plot aim points
    print("PLOTTING AIM POINTS")
    aim_points = np.loadtxt("./parallel_test/aim_points.txt")
    for idx, apt in enumerate(aim_points):
        ax.scatter(*apt, marker='x', color='k', s=25) #, label="AIM POINT {}".format(idx))"""

    # Plot response path
    print("PLOTTING RESPONSE")
    r_x = np.loadtxt("./parallel_test/response_x.txt")
    r_y = np.loadtxt("./parallel_test/response_y.txt")
    r_z = np.loadtxt("./parallel_test/response_z.txt")
    ax.plot(r_x, r_y, r_z, '--', color='blue', alpha=0.5, label="RESPONSE ROCKET PATH")

    # Check if successful interception
    first_detection = int(np.loadtxt("./parallel_test/first_detection.txt"))

    print("FIRST DETECTION TIME INDEX: {}".format(first_detection))
    dist_to_enemy = []
    print(ex[first_detection], ey[first_detection], ez[first_detection])
    for i, j in enumerate(r_x):
        offset_i = i + first_detection
        r_vec = np.array([r_x[i], r_y[i], r_z[i]])
        e_vec = np.array([ex[offset_i], ey[offset_i], ez[offset_i]])
        dist_to_enemy.append(np.linalg.norm(r_vec - e_vec))
    print(dist_to_enemy)
    hit_index = np.argmin(np.array(dist_to_enemy))
    print(len(dist_to_enemy))
    print(f"CLOSEST DISTANCE: {min(dist_to_enemy)} at index {hit_index}")

    print(f'{ex[hit_index+ first_detection]} {ey[hit_index+ first_detection]} {ez[hit_index+ first_detection]}')
    print(f'{r_x[hit_index]} {r_y[hit_index]} {r_z[hit_index]}')

    ax.plot(ex[:hit_index+first_detection], ey[:hit_index+first_detection], ez[:hit_index+first_detection], color='r', label="ENEMY ROCKET")
    ax.plot(r_x[:hit_index], r_y[:hit_index], r_z[:hit_index], color='blue', label="RESPONSE ROCKET")


    # Plot detection hemisphere
    print("PLOTTING DOME")
    radius = 1500
    origin = [2000, 2000]
    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j]
    dome_x = radius*np.sin(phi)*np.cos(theta) + origin[0]
    dome_y = radius*np.sin(phi)*np.sin(theta) + origin[1]
    dome_z = radius*np.cos(phi)
    ax.plot_surface(dome_x, dome_y, dome_z, alpha=0.15, color='g')

    ground_theta = np.linspace(0, 2 * np.pi, 100)
    ground_x = radius * np.cos(ground_theta) + origin[0]
    ground_y = radius * np.sin(ground_theta) + origin[1]
    ax.plot(ground_x, ground_y, alpha=0.5, color='g')
    #ax.set(zlim=(0, max(max(ez), max(pz))))
    ax.set(zlim=(0, max(ez)))

    ax.legend(loc='best')
    plt.show()


def test_enemy_and_response():
    directory = './parallel_test'
    if not os.path.exists(directory):
        os.makedirs(directory)

    #delta_t = 0.025
    p1 = Process(target=enemy_rocket)
    p1.start()
    p2 = Process(target=response_rocket)
    p2.start()
    p1.join()
    p2.join()


if __name__ == '__main__':
    #test_enemy_and_response()
    plot_enemy_projection()
    plot2()

