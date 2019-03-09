import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Dense
import numpy as np
import keras
class Predictor(object):


    def __init__(self, url_results, url_features):
        X = []
        Y = []
        f = open(url_results, 'r')
        line = f.readline()
        while line:
            arch = line.split(" ")
            if len(arch)>1:
                Y.append(arch[1])
            line = f.readline()
        f.close()

        print(len(Y))

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


        num_classes = 100
        nb_elements = len(X)
        split_elements = int(9*nb_elements/10)

        X = np.array(X).astype(int)

        #We split the Y values into 10 batchs
        Y = np.floor(np.array(Y).astype(float) *num_classes).astype(int)
        print(Y[0])

        Y =  keras.utils.to_categorical(Y, num_classes)

        model = Sequential()
        model.add(Dense(128, input_dim=8847, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="sgd")

        X_train = X[0:split_elements]
        Y_train = Y[0:split_elements]
        model.fit(X_train, Y_train, epochs=100, verbose=1)
        score = model.evaluate(X[split_elements:], Y[split_elements:])

        print("predictor score {0}".format(score))

        
        print("predict {0} vs {1}".format(model.predict(X[split_elements:]), Y[split_elements:]))
        

 
if __name__ == '__main__':
    
    predictor = Predictor("./report_100.txt", "./100Products.pdt")