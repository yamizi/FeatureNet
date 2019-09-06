import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os
from mpl_toolkits.mplot3d import Axes3D

def load_results(url_results):
    Y = []
    f = open(url_results, 'r')
    line = f.readline()
    while line:
        arch = line.split(":")
        if len(arch)>1:
            #print(arch[0])
            Y.append(json.loads(arch[1]))
        line = f.readline()
    f.close()

    return Y

def get_fronts(df):
    nb_elements = len(df["accuracy"])
    df["dom"] = [0 for i in range(nb_elements)]
    dominates = []
    fronts = [[]]

    # finding the first front
    for i in range(nb_elements):
        dominates.append([])
        isDominant = True
        for j in range(nb_elements):
            if i == j:
                continue
            # if i dominates j
            if df["accuracy"][i] > df["accuracy"][j] and df["robustness"][i] > df["robustness"][j]:
                dominates[i].append(j)
            # else if i is dominated by j
            elif  df["accuracy"][j] > df["accuracy"][i] and df["robustness"][j] > df["robustness"][i]:
                df['dom'][i] += 1
        if df['dom'][i] == 0:
            fronts[0].append(i)

    return fronts[0]

def pareto(url_results="../products/cifar_12_0.1_0.1/20products_e{}.json", nb_epochs=0, name=""):
    all_set = []

    if nb_epochs:
        for i in range(nb_epochs):
            y = load_results(url_results.format(i))
            all_set.insert(0,y)
    
    else:
        i = 0;
        exists = True
        while(exists):
            filename = url_results.format(i)
            if not os.path.isfile(filename):
                exists = False
                #print("stop {}".format(filename))
                continue

            #print("load {}".format(filename))
            y = load_results(filename)
            all_set.insert(0,y)
            i = i+1

    nb_epochs = len(all_set)
    fig = plt.figure()
    skip = np.ceil(nb_epochs / 10)

    for i in range(nb_epochs):
        if i%skip!=0:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Robustness')

        Y = all_set[i]
        Y = [y for y in Y if y[0]>0.1]
        accuracy = np.array([float(y[0]) for y in Y])
        robustness = np.array([float(y[1][4]) for y in Y])
        ax.scatter(accuracy,robustness)

        front = get_fronts({"accuracy":accuracy, "robustness":robustness})
        if len(front)>1:
            x_front = [val for i,val in enumerate(accuracy) if i in front]
            y_front = [val for i,val in enumerate(robustness) if i in front]
            ax.scatter(x_front,y_front,c="red")
            ax.set_ylim([min(robustness)*0.999, max(robustness)*1.001])
        
        plt.title("{} epoch {}".format(name,i))
    plt.show()

def histo(url_results="../products/cifar_12_0.1_0.1/20products_e{}.json", nb_epochs=0, name="", show_max_accuracy=True, show_max_robustness=False):
    
    handles = []
    all_set = []
    if nb_epochs:
        for i in range(nb_epochs):
            y = load_results(url_results.format(i))
            all_set.insert(0,y)
    
    else:
        i = 0;
        exists = True
        while(exists):
            filename = url_results.format(i)
            if not os.path.isfile(filename):
                exists = False
                #print("stop {}".format(filename))
                continue
            y = load_results(filename)
            all_set.insert(0,y)
            i = i+1

    robustness_metrics = ["fgsm", "pgd", "cw"]
    robustness_linestyle=["--",".","x",""]
    robustness_markers=["o","v","^","."]
    nb_epochs = len(all_set)
    #print(nb_epochs)

    
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('CW Robustness')
    #ax.set_zlabel('Size')

    colors = plt.get_cmap("Set1")
    #print(colors)
    color_index = 0

    skip = np.ceil(nb_epochs / 10)
    #skip = 1

    full_acc =  np.array([])
    full_rob =  np.array([])
    full_rob_pgd = np.array([])
    full_rob_cw = np.array([])

    for k in range(nb_epochs):
        i = nb_epochs - k -1-1
        Y = all_set[i]
        if i%skip!=0:
            continue
        Y = [y for y in Y if y[0]>0.5]

        if len(Y)==0:
            continue

        accuracy = np.array([float(y[0]) for y in Y])
        full_acc = np.concatenate((full_acc, accuracy))

        s = len(Y[0][1])
        if len(Y[0][1])>3:
            robustness = np.array([[float(y[1][4]) for y in Y]])
        else:
            rob = [y[1][2][1] for y in Y]
            robustness =  np.array([[r for r in o if type(r) is list] for o in rob])
            # real_accuracy = np.array([o[1] for o in rob if type(o) is list])
            # adv_accuracy = np.array([o[2] for o in rob if type(o) is list])
        
        full_rob = np.concatenate((full_rob, robustness[:,0,0]))
        full_rob_pgd = np.concatenate((full_rob_pgd, robustness[:,0,0]))
        full_rob_cw = np.concatenate((full_rob_cw, robustness[:,1,0]))

        nb_metrics = 1 #robustness.shape[1]
        for j in range(nb_metrics):
            
            color = colors.colors[color_index] if color_index< len(colors.colors) else np.random.rand(3,) 
            
            robustness_metric = robustness[:,j,0]
            ax.scatter(accuracy, robustness_metric, c=color, marker=robustness_markers[j])
            if show_max_accuracy:
                accuracy_mean =accuracy.mean()
                ax.axvline(x=accuracy_mean, label='Max accuracy', linestyle='--', color=color)
            if show_max_robustness:
                robustness_mean =robustness_metric.mean()
                ax.axhline(y=robustness_mean, label='Max robustness', linestyle='--', color=color)
        
        #ax.set_ylim([min(robustness)*0.99, max(robustness)*1.1])
        handles.append(mpatches.Patch(color=color, label='Iteration {}'.format(k)))
        color_index = color_index+1

    fit = 1
    p20 = np.poly1d(np.polyfit(full_acc, full_rob_pgd, fit))
    # p30 = np.poly1d(np.polyfit(full_acc, full_rob_cw, fit))

    xp = np.linspace(min(full_acc), max(full_acc), 100)
    # plt.plot(xp, p10(xp), 'k-')
    plt.plot(xp, p20(xp), 'k-')
    # plt.plot(xp, p30(xp), 'k-')

    plt.legend(handles=handles, frameon=False)
    plt.title(name)

#histo("../products/run25a/cifar_MutationStrategies.ALL/25_0.1_0.2/50products_e{}.json",13)
#histo("../products/run24a/cifar_25_0.1_0.2/50products_e{}.json")
#histo("../products/local/cifar/1_0.1_0.1_1561474651 = complete/5products_e{}.json",20)
#histo("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561488972 = 33epochs/e{}.json",35)
#histo("../products/run26c/cifar/ee50_te12_mr0.1_sr0.1_1561501150/e{}.json",5)
#histo("../products/run26a/cifar/ee50_te12_mr0.1_sr0.1/e{}.json",3)

#histo("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561543665 = 13 choice/e{}.json",14, "with choice mutation",show_max_accuracy=False, show_max_robustness=True)
#histo("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561543665 = 13 choice/e{}.json",10, "with choice mutation",show_max_accuracy=True, show_max_robustness=False)
#histo("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561559187/e{}.json",9, "with all mutations")

#pareto("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561543665 = 13/e{}.json",13)
#pareto("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561488972 = 33epochs/e{}.json",35)

#histo("../products/local/cifar/ee70_te1_mr0.1_sr0.1_1561562201 = 35 all/e{}.json")
#pareto("../products/local_cp/cifar/ee70_te12_mr0.1_sr0.1/e{}.json", name="Pareto Choice")

#histo("../products/local_cp/cifar/ee70_te12_mr0.1_sr0.1/e{}.json", 10, name="Pareto Choice",show_max_accuracy=True, show_max_robustness=False)
#histo("../products/local_cp/cifar/ee70_te12_mr0.1_sr0.1/e{}.json", 14, name="Pareto Choice",show_max_accuracy=False, show_max_robustness=True)

#histo("../products/run27a/cifar/ee50_te300_mr0.1_sr0.1/e{}.json", name="run27 All")
#histo("../products/run27c/cifar/ee50_te300_mr0.1_sr0.1/e{}.json", name="run27 Choice")

# histo("../products/run28a/cifar/ee50_te300_mr0.1_sr0.1/e{}.json", name="run28 All")
#histo("../products/run28c/cifar/ee50_te300_mr0.1_sr0.1/e{}.json", name="run28 Choice")

# histo("../products/run29a/cifar/ee50_te300_mr0.1_sr0.1/e{}.json", name="run29 All FGSM")
#histo("../products/run29c/cifar/ee50_te300_mr0.1_sr0.1/e{}.json", name="run29 Choice FGSM")

#histo("../products/run30c/cifar/ee50_te150_mr0.1_sr0.1_1562054644/e{}.json", name="run30 Diversity")

#histo("../products/local_cp/cifar/ee70_te12_mr0.1_sr0.1_1561843599/e{}.json", 12, name="Pareto Choice no Pooling")

#histo("../products/cifar/ee50_te12_mr0.1_sr0.1_1561987431/e{}.json", 0, name="Pareto Choice no Pooling Diversity initial")

#histo("../products/pledge/cifar/ee50_te12_mr0.1_sr0.5/e{}.json", 0, name="Pareto Choice no Pooling Diversity")

#histo("../products/pledge_loss/cifar/ee50_te50_mr0.1_sr0.2_1562407408/e{}.json", 10, name="Pareto Choice no Pooling LeNet 50e loss")

#histo("../products/pledge_loss/cifar/ee50_te12_mr0.1_sr0.2_1562923033/base.json", name="Initial PLEDGE population")

#histo("../products/pledge_loss/cifar/ee50_te12_mr0.1_sr0.2_1562923033/e{}.json", name="Initial PLEDGE population evol")


#histo("../products/2metrics\cifar#lenet5\ee10_te12_mr0.1_sr0.2/e{}.json", name="3 metrics", show_max_accuracy=False, show_max_robustness=False)

#histo("../products/2metrics\cifar#lenet5\ee10_te12_mr0.1_sr0.2/e{}.json", name="3 metrics", show_max_accuracy=False, show_max_robustness=False)

histo("../products/2metrics\cifar#keras\ee10_te50_mr0.1_sr0.2_1566924825/e{}.json", name="Robustness driven sampling", show_max_accuracy=False, show_max_robustness=False)


plt.show()
