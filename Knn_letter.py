from collections import Counter

import numpy as np, pandas as pd
from sklearn.model_selection import KFold 

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k):
        self.k = k
        #return df

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
    letter_dataset = pd.read_csv('./data/letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
    ObjectColumns = letter_dataset.select_dtypes(include=np.object).columns.tolist()
    letter_dataset['lettr'] = [ord(item)-64 for item in letter_dataset['lettr']]
    for i in range(10):
        letter_dataset = letter_dataset.sample(frac=1)
        k = 2
        
        model = KNN(k)
        X = letter_dataset.drop(["id","decision"], axis = 1)
        y = letter_dataset.decision
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