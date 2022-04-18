import math, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class KNN:
    def pre_processing(self, df):

        car_dataset["buying"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace = True)
        car_dataset["maint"].replace(["vhigh", "high", "med", "low"], [3, 2, 1, 0], inplace = True)
        car_dataset["doors"].replace(["5more"], [5], inplace = True)
        car_dataset["persons"].replace(["more"], [4], inplace = True)
        car_dataset["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace = True)
        car_dataset["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace = True)
        car_dataset["decision"].replace(["unacc", "acc", "good", "vgood"], [1, 0, 2, 3], inplace = True)
        
        car_dataset['buying'] = car_dataset['buying'].astype(int)
        car_dataset['maint'] = car_dataset['maint'].astype(int)
        car_dataset['doors'] = car_dataset['doors'].astype(int)
        car_dataset['persons'] = car_dataset['persons'].astype(int)
        car_dataset['lug_boot'] = car_dataset['lug_boot'].astype(int)
        car_dataset['safety'] = car_dataset['safety'].astype(int)
        car_dataset['decision'] = car_dataset['decision'].astype(int)
        
        return df
        
    def knn_implementation(self, x_train, y_train, x_test, K):
        dists, train_size = {}, len(x_train)
        for i in range(train_size):
            d = self.euclidian_dist(x_train.iloc[i], x_test)
            dists[i] = d
    
        k_neighbors = sorted(dists, key=dists.get)[:K]
    
        qty_labels = [0,0,0,0]
        for index in k_neighbors:
            if int(y_train.iloc[index]) == 0:
                qty_labels[0] += 1
            elif int(y_train.iloc[index]) == 1:
                qty_labels[1] += 1
            elif int(y_train.iloc[index]) == 2:
                qty_labels[2] += 1
            elif int(y_train.iloc[index]) == 3:
                qty_labels[3] += 1
            
        max_label = max(qty_labels)
        return qty_labels.index(max_label)

        

    def euclidian_dist(self, x_train, x_test):
        dim, sum_ = len(x_train), 0
        for index in range(dim - 1):
            sum_ += math.pow(x_train[index] - x_test[index], 2)
        return math.sqrt(sum_)

    def mean_accuracy(self, accuracy):
        return sum(accuracy)/ len(accuracy)
        

if __name__ == "__main__":

    car_dataset = pd.read_csv("./data/car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
    accuracy = []
    for i in range(10):
    # Shuffle the dataset 
        car_dataset = car_dataset.sample(frac=1)
        knn = KNN()
        df = knn.pre_processing(car_dataset)
        X = df.drop(["decision"], axis = 1)
        y = df.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        correct, K = 0, 15
        for i in range(0, len(X_test)):
            #print(x_train.iloc[i])
            label = knn.knn_implementation(X_train, y_train, X_test.iloc[i], K)
        if int(y_test.iloc[i]) == label:
                correct += 1
        accuracy.append(100 * correct / len(X_train))
    mean = knn.mean_accuracy(accuracy)
    print("Mean Accuracy of 10 folds: %.2f%%" % mean)
          