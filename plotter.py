import matplotlib.pyplot as plt
import numpy as np

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

    f = open(url_results, 'r')
    line = f.readline()
    while line:
        arch = line.split(" ")
        if len(arch)>1:
            Y.append(arch)
        line = f.readline()
    f.close()

    print(len(Y))

    #We filter only architectures where accuracy >0.8
    Y = [y for y in Y if float(y[1]) >= 0.8]

    #We filter only architectures < 200K parameters
    Y = [y for y in Y if int(y[4]) < 50000]
    print(Y)

    nb_elements = len(Y)
    print(nb_elements)
    
    accuracy = np.array([float(y[1]) for y in Y])
    training_time = np.array([float(y[3]) for y in Y])
    nb_params = np.array([float(y[4]) for y in Y])
    #plt.xticks(np.arange(0, nb_elements, 1.0))

    fig, ax1 = plt.subplots()
    ax1.set_title(plot_prefix)
    ax1.plot(accuracy,'bx')
    ax1.set_ylabel('accuracy', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(training_time, 'r-')
    ax2.set_ylabel('training time (s)', color='r')
    ax2.tick_params('y', colors='r')
    

    fig2, ax21 = plt.subplots()
    ax21.set_title(plot_prefix)
    ax21.plot(accuracy,'bx')
    ax21.set_ylabel('accuracy', color='b')
    ax21.tick_params('y', colors='b')
    ax22 = ax21.twinx()
    ax22.plot(np.log(nb_params), 'g-')
    ax22.set_ylabel('nb parameters (log)', color='g')


    fig3, ax31 = plt.subplots()
    ax31.set_title(plot_prefix)
    ax31.plot(accuracy,'bx')
    ax31.set_ylabel('accuracy', color='b')
    ax31.tick_params('y', colors='b')
    ax32 = ax31.twinx()
    ax32.plot(accuracy/nb_params*1000000, 'g-')
    ax32.set_ylabel('accuracy / parameters(M)', color='g')


    plt.savefig("{0}_{1}_accuracy".format(plot_path, plot_prefix))
    plt.show()


if __name__ == '__main__':
    
    #plot("./report_1000Products_cifar.txt", "./1000Products.pdt", "1000_cifar")
    plot("./report_1000Products_mnist.txt", "./1000Products.pdt", "1000_mnist")