
from scipy import stats
import numpy as np

def load_scores(url_scores):
    Y = []
    f = open(url_scores, 'r')
    line = f.readline()
    while line:
        arch = line.split(" ")
        if len(arch)>1:
            Y.append(float(arch[1]))
        line = f.readline()
    f.close()

    return Y

mnist = load_scores("report1000Products_mnist.txt")
cifar = load_scores("report1000Products_cifar.txt")


def get_correlations(mnist, cifar):
    tau, p_value = stats.kendalltau(mnist, cifar)
    (pearson,p_p) = stats.pearsonr(mnist, cifar)
    (spearman,p_s) = stats.spearmanr(mnist, cifar)
    print(tau, pearson, spearman)


#get_correlations(mnist, cifar)


def split_sets(accuracies, threshold):
    accuracies = np.array(accuracies)
    s = np.where( accuracies > threshold )
    
    return s[0].size

mnist_split = [split_sets(mnist, 0.3), split_sets(mnist, 0.5), split_sets(mnist, 0.9), split_sets(mnist, 0.95)]
cifar_split = [split_sets(cifar, 0.3), split_sets(cifar, 0.5), split_sets(cifar, 0.9), split_sets(cifar, 0.95)]
print(cifar_split)