import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D

def load_results(url_results):
    Y = []
    f = open(url_results, 'r')
    line = f.readline()
    while line:
        arch = line.split(":")
        if len(arch)>1:
            Y.append(json.loads(arch[1]))
        line = f.readline()
    f.close()

    return Y

def histo(url_results="../../products/cifar/20products_e{}.json"):
    
    handles = []
    all_set = []
    for i in range(4):
        y = load_results(url_results.format(i))
        all_set.append(y)

    print(len(all_set))

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Robustness')
    ax.set_zlabel('Size')

    for i,Y in enumerate(all_set):
        Y = [y for y in Y if y[0]>0.1]
        accuracy = np.array([float(y[0]) for y in Y])
        robustness = np.array([float(y[1][4]) for y in Y])
        size = np.array([float(y[1][2]) for y in Y])
        color = np.random.rand(3,)
        ax.scatter(accuracy, robustness,size, c=color)
        handles.append(mpatches.Patch(color=color, label='Iteration {}'.format(i+1)))

    plt.legend(handles=handles, frameon=False)

histo()
plt.show()