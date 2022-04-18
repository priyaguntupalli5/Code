from collections import Counter

import numpy as np, pandas as pd
from sklearn.model_selection import KFold 

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k):
        self.k = k
        #return mushroom_dataset

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    mushroom_dataset = pd.read_csv("./data/mushroom.data", names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment", 
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", 
    "population", "habitat"])
    k = 2

    mushroom_dataset.replace(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],inplace = True)
    
    mushroom_dataset["stalk-root"].replace(["?"], ['0'], inplace = True)
    mushroom_dataset["stalk-root"] = mushroom_dataset['stalk-root'].astype(int)
    average = mushroom_dataset["stalk-root"].mean()
    mushroom_dataset["stalk-root"].replace(["?"], [average], inplace = True)

    mushroom_dataset["decision"] = mushroom_dataset['decision'].astype(int)
    mushroom_dataset["cap-shape"] = mushroom_dataset['cap-shape'].astype(int)
    mushroom_dataset["cap-surface"] = mushroom_dataset['cap-surface'].astype(int)
    mushroom_dataset["cap-color"] = mushroom_dataset['cap-color'].astype(int)
    mushroom_dataset["bruises"] = mushroom_dataset['bruises'].astype(int)
    mushroom_dataset["odor"] = mushroom_dataset['odor'].astype(int)
    mushroom_dataset["gill-attachment"] = mushroom_dataset['gill-attachment'].astype(int)
    mushroom_dataset["gill-spacing"] = mushroom_dataset['gill-spacing'].astype(int)
    mushroom_dataset["gill-size"] = mushroom_dataset['gill-size'].astype(int)
    mushroom_dataset["gill-color"] = mushroom_dataset['gill-color'].astype(int)
    mushroom_dataset["stalk-shape"] = mushroom_dataset['stalk-shape'].astype(int)
    mushroom_dataset["stalk-surface-above-ring"] = mushroom_dataset['stalk-surface-above-ring'].astype(int)
    mushroom_dataset["stalk-surface-below-ring"] = mushroom_dataset['stalk-surface-below-ring'].astype(int)
    mushroom_dataset["stalk-color-above-ring"] = mushroom_dataset['stalk-color-above-ring'].astype(int)
    mushroom_dataset["stalk-color-below-ring"] = mushroom_dataset['stalk-color-below-ring'].astype(int)
    mushroom_dataset["veil-type"] = mushroom_dataset['veil-type'].astype(int)
    mushroom_dataset["veil-color"] = mushroom_dataset['veil-color'].astype(int)
    mushroom_dataset["ring-number"] = mushroom_dataset['ring-number'].astype(int)
    mushroom_dataset["ring-type"] = mushroom_dataset['ring-type'].astype(int)
    mushroom_dataset["spore-print-color"] = mushroom_dataset['spore-print-color'].astype(int)
    mushroom_dataset["population"] = mushroom_dataset['population'].astype(int)
    mushroom_dataset["habitat"] = mushroom_dataset['habitat'].astype(int)
    acc=[]
    
    for i in range(10):
        mushroom_dataset = mushroom_dataset.sample(frac=1)
        k = 2
        
        model = KNN(k)
        X = mushroom_dataset.drop(["decision"], axis = 1)
        y = mushroom_dataset.decision
        X = X.to_numpy()
        y = y.to_numpy()

        kf = KFold(n_splits=5)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X[train_index,:],X[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
     
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
     
            acc.append(accuracy_score(pred_values , y_test))

    mean = sum(acc)/len(acc)

    print("KNN classification accuracy", mean)