import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Dense
import numpy as np
import keras
class Predictor(object):


    def __init__(self, url_results, url_features, url_tests=None):
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
        Y_batch = np.floor(np.array(Y).astype(float) *num_classes).astype(int)

        Y =  keras.utils.to_categorical(Y_batch, num_classes)

        model = Sequential()
        model.add(Dense(32, input_dim=X[0].size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="sgd")

        if url_tests:
            X_train = X
            Y_train = Y
            X_t = []
            Y_t = []
            f = open(url_results, 'r')
            line = f.readline()
            while line:
                arch = line.split(" ")
                if len(arch)>1:
                    Y_t.append(float(arch[1]))
                line = f.readline()
            f.close()

            f = open(url_features, 'r')
            line = f.readline()
            while line:
                
                product_features = line.split(";")
                
                if len(product_features) > 1:
                    product_features = product_features[:len(product_features)-1]
                    product_features = [int(x) for x in product_features]
                    product_features = [0 if x<0 else 1 for x in product_features]
                    X_t.append(product_features) 

                line = f.readline()
            f.close()

            Y_test =  keras.utils.to_categorical(np.floor(np.array(Y_t).astype(float) *num_classes).astype(int), num_classes)
            X_test = X_t
        
        else:
            X_train = X[0:split_elements]
            Y_train = Y[0:split_elements]
            X_test = X[split_elements:]
            Y_test =  Y[split_elements:]

        model.fit(X_train, Y_train, epochs=100, verbose=1)
        score = model.evaluate(X_test, Y_test)

        prediction = model.predict(X_test)
        print(np.argmax(prediction, axis=1),np.argmax(Y_test, axis=1))

        #top = np.amax(prediction, axis=1)
        #top_index = [np.where(prediction[index] == top_e)[0] for (index,top_e) in enumerate(top)]
        #print(top, top_index)
        print("predictor score {0}".format(score))      
       # print("predict {0} {1}".format(np.argmax(prediction, axis=1), Y[split_elements:]))
        

 
if __name__ == '__main__':
    
    #predictor = Predictor("./reports/report_100.txt", "./100Products.pdt")
    predictor = Predictor("./report1000Products_cifar.txt", "./1000Products.pdt")
    