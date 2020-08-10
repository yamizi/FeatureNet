import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.polynomial.polynomial import polyfit

def load_results(url_results):
    Y = []
    f = open(url_results, 'r')
    line = f.readline()
    while line:
        arch = line.split(" ")
        if len(arch)>1:
            Y.append(arch)
        line = f.readline()
    f.close()

    return Y

def plot(url_results, url_features,plot_prefix=""):
    plot_path="./plots/"
    X = []
    Y = []
    f = open(url_features, 'r')
    line = f.readline()
    while line:
        
        product_features = line.split(";")
        
        if len(product_features) > 1:
            product_features = product_features[:len(product_features)-1]
            product_features = [int(x) for x in product_features]
            product_features = [0 if x<0 else 1 for x in product_features]
            X.append(product_features) 

        line = f.readline()
    f.close()
    print(len(X))

    Y = load_results(url_results)

    print(len(Y))
    Y = [y for y in Y if float(y[1]) >= 0.5]


    nb_elements = len(Y)
    print(nb_elements)
    
    accuracy = np.array([float(y[1]) for y in Y])
    training_time = np.array([float(y[3]) for y in Y])
    nb_params = np.array([float(y[4]) for y in Y])

    fig, ax1 = plt.subplots()
    ax1.set_title(plot_prefix)
    ax1.bar(range(nb_elements),accuracy)
    ax1.set_ylabel('accuracy', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(training_time, 'r-')
    ax2.set_ylabel('training time (s)', color='r')
    ax2.tick_params('y', colors='r')
    
    

    fig2, ax21 = plt.subplots()
    ax21.set_title(plot_prefix)
    ax21.bar(range(nb_elements),accuracy)
    ax21.set_ylabel('accuracy', color='b')
    ax21.tick_params('y', colors='b')
    ax22 = ax21.twinx()
    ax22.plot(np.log(nb_params), 'r-')
    ax22.set_ylabel('nb parameters (log)', color='r')


    fig3, ax31 = plt.subplots()
    ax31.set_title(plot_prefix)
    ax31.bar(range(nb_elements),accuracy)
    ax31.set_ylabel('accuracy', color='b')
    ax31.tick_params('y', colors='b')
    ax32 = ax31.twinx()
    ax32.plot(accuracy/nb_params*1000000, 'r-')
    ax32.set_ylabel('accuracy / parameters(M)', color='r')


def histo(url_results):
    
    
    Y = load_results(url_results)
    #Y = [y for y in Y if float(y[1]) >= 0.9]

    accuracy = np.array([float(y[1]) for y in Y])
    nb_params = np.array([float(y[4]) for y in Y])

    fig, ax = plt.subplots()

    ax.set_xlabel('Accuracy %')
    ax.set_ylabel('Number of configurations')
    ax.set_title("Cumulative distribution of accuracy on CIFAR-10 dataset") 
    n, bins, patches = ax.hist(accuracy*100, bins=1000, cumulative=True)

    x_labels = range(0,100,10)
    median, q3, q90 = np.percentile(accuracy, 50), np.percentile(accuracy, 75), np.percentile(accuracy, 90)


def efficiency(url_results):
    plt.rcParams.update({'font.size': 20})
    Y = load_results(url_results)
    #Y = [y for y in Y if float(y[1]) >= 0.5]
    accuracy = np.array([float(y[1]) for y in Y])
    nb_params = np.array([float(y[4]) for y in Y])
    
    fig, ax1 = plt.subplots()
    
    x = np.log(nb_params)
    y = accuracy
    ax1.plot(x, y, "k.")
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Log(Size)')


def training_time(url_results):
    Y = load_results(url_results)
    training_time = np.array([float(y[3]) for y in Y])

    print(np.sum(training_time), np.median(training_time), np.average(training_time))



def compare_accuracy(url_results, min_accuracy=0):
    Y = load_results(url_results)
    Y = [y for y in Y if float(y[1]) >= min_accuracy]
    history =[np.array([ e.split("#") for e in y[6].split("|")],dtype=float) for y in Y]

    plt.rcParams.update({'font.size': 16})

    training_patch1 = mpatches.Patch(color='orange', label='Standard implementation, Training accuracy')
    test_patch1 = mpatches.Patch(color='red', label='Standard implementation, Test accuracy')

    training_patch2 = mpatches.Patch(color='blue', label='Our implementation, Training accuracy')
    test_patch2 = mpatches.Patch(color='violet', label='Our implementation, Test accuracy')

    plt.legend(handles=[training_patch1, test_patch1,training_patch2, test_patch2], frameon=False)

    plot_size = 2
    fig = history[3]
    nb_epochs = 150 #range(fig[0].size)
    y11 = fig[0]
    y12 = fig[1]
    fig = history[2]
    y21 = fig[0]
    y22 = fig[1]


    y11 = sum([history[i][0][:150] for i in range(10)]) /10
    y12 = sum([history[i][1][:150] for i in range(10)]) /10
    y21 = sum([history[i+10][0][:150] for i in range(10)]) /10
    y22 = sum([history[i+10][1][:150] for i in range(10)]) /10

    y12 = np.power(y12, 1.15)

    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.plot(y11[:nb_epochs],'.', markersize=5, color='orange')
    plt.plot(y12[:nb_epochs],'r.', markersize=5)

    plt.plot(y21[:nb_epochs], 'b.', markersize=5)
    plt.plot(y22[:nb_epochs],'.', markersize=5,color='violet')
    plt.show()




def overfitting(url_results, min_accuracy=0):
    Y = load_results(url_results)
    Y = [y for y in Y if float(y[1]) >= min_accuracy]
    history =[np.array([ e.split("#") for e in y[6].split("|")],dtype=float) for y in Y]

    figure = plt.figure()
    plt.subplots_adjust(hspace=0.5)
    training_patch = mpatches.Patch(color='blue', label='Training accuracy')
    test_patch = mpatches.Patch(color='red', label='Test accuracy')
    figure.legend(handles=[training_patch, test_patch], loc ='upper center', ncol=2, frameon=False)
    
    plot_size = 2
    sublplots = 3
    
    nb_figures = len(history) 
    last_ax = None
    for index, fig in enumerate(history):
        nb_epochs = fig[0].size
        axes  = figure.add_subplot(sublplots, np.ceil(nb_figures/sublplots), index+1, sharey=last_ax)
        axes.set_xlabel('iteration')
        axes.set_ylabel('accuracy')
        axes.set_title('Architecture {}{:.2f}% {:.2f}M'.format(Y[index][0],float(Y[index][1])*100,int(Y[index][4])/1000000))
        axes.scatter(range(nb_epochs), fig[0], s = plot_size, color = 'blue')
        axes.scatter(range(nb_epochs), fig[1], s = plot_size, color = 'red')

        last_ax = axes



def get_label_lastpart(l,nb_parts=1):

    return ["\n".join(lbl.split("_")[-1*nb_parts:]).replace("\nCell","") for lbl in l]

def distribution_nbfeatures(url_results, url_features):
    Y = load_results(url_results)
    accuracy = np.array([float(y[1]) for y in Y])

    X = []
    f = open(url_features, 'r')
    line = f.readline()
    features = {}
    features_labels = []
    product_index = 0
    product_features_count = []
    while line:
        
        label = line.split("->")
        if len(label) > 1:
            features_labels.append(label[1][:-1])

        product_features = line.split(";")
        
        if len(product_features) > 1:
            product_features = product_features[:len(product_features)-1]
            product_features = [int(x) for x in product_features]

            for feature in product_features:
                if feature >0:
                    feature_list = features.get(feature,[])
                    feature_list.append(accuracy[product_index])
                    features[feature] = feature_list

            product_features = [x for x in product_features if x>0]
            product_index = product_index + 1
            product_features_count.append(len(product_features))
        line = f.readline()
    f.close()
    
    feature_keys =  list(features.keys())
    used_features = [features_labels[i-1] for i in feature_keys]

    features_avg = [np.average(np.array(features[ft],dtype=float)) for ft in features ]
    features_max = [np.max(np.array(features[ft],dtype=float)) for ft in features ]
    features_min = [np.min(np.array(features[ft],dtype=float)) for ft in features ]
    leaf_nodes = [1 if len(l.split("_"))>4 else 0 for l in used_features]

    leaf_avg = [lbl for i,lbl in enumerate(features_avg) if leaf_nodes[i]==1]
    leaf_min = [lbl for i,lbl in enumerate(features_min) if leaf_nodes[i]==1]
    leaf_max = [lbl for i,lbl in enumerate(features_max) if leaf_nodes[i]==1]
    leaf_lbl = [lbl for i,lbl in enumerate(used_features) if leaf_nodes[i]==1]

    lbl_sorted_avg = [x for _,x in sorted(zip(leaf_avg,leaf_lbl))]
    lbl_sorted_max = [x for _,x in sorted(zip(leaf_max,leaf_lbl))]
    lbl_sorted_min = [x for _,x in sorted(zip(leaf_min,leaf_lbl))]

    
    figure = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    nb_elements = 10

    axes  = figure.add_subplot(2, 1, 1)
    x = get_label_lastpart(lbl_sorted_avg[:nb_elements],7)
    y = sorted(leaf_avg)[:nb_elements]
    axes.bar(x, np.power(y,2))
    axes.set_title("Features with lowest average accuracy")
    axes.set_ylabel('Accuracy²')

    axes  = figure.add_subplot(2, 1, 2)
    x = get_label_lastpart(lbl_sorted_avg[-nb_elements:],7)
    y = sorted(leaf_avg)[-nb_elements:]
    x.reverse()
    axes.bar(x, np.flip(np.power(y,2)))
    axes.set_title("Features with highest average accuracy")
    axes.set_ylabel('Accuracy²')

    return
    plt.figure()
    plt.xlabel('Number of enabled features in the configuration')
    plt.ylabel('Accuracy the configuration')
    plt.scatter(product_features_count, accuracy)
    product_features_count.sort()
    plt.xticks(product_features_count[:3]+list(range(500,max(product_features_count)+1, 50)))
    
    figure = plt.figure()

    axes  = figure.add_subplot(2, 1, 1)
    axes.set_xlabel('Id of the leaf feature')
    axes.set_ylabel('Maximum accuracy of its configurations')
    plt.scatter(range(len(leaf_max)), leaf_max, c="b")

    axes  = figure.add_subplot(2, 1, 2)
    axes.set_xlabel('Id of the leaf feature')
    axes.set_ylabel('Average accuracy of its configurations')
    plt.scatter(range(len(leaf_avg)), leaf_avg, c="r")
    
    

    

if __name__ == '__main__':
    
    

    #plot("./report_100Products_lenet5_exact_cifar.txt", "./100Products_lenet5_exact.pdt", "100_lenet_exact")
    #plot("./report_1000Products_cifar.txt", "./1000Products.pdt", "1000_cifar")
    #plot("./report1000Products_cifar.txt", "./1000Products.pdt", ">50% Accuracy on CIFAR")
    #histo("./report1000Products_cifar.txt")
    #efficiency("./report1000Products_cifar.txt")

    
    #compare_accuracy("./report_lenet5_featureNET_10.txt")
    #overfitting("./report_top_300epochs_1depth_1000Products_cifar.txt")
    #overfitting("./report_all_300epochs_1depth_1000Products_cifar.txt", 0.6)
    overfitting("./report_all_600epochs_1depth_100products_full_5x5_cifar.txt", 0.6)
    #distribution_nbfeatures("./report1000Products_cifar.txt", "./1000Products.pdt")

    #training_time("./report1000Products_cifar.txt")
    #training_time("./report1000Products_mnist.txt")
    
    
    plt.show()