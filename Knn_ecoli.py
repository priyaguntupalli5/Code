from collections import Counter

import numpy as np, pandas as pd


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k):
        self.k = k
        #return ecoli_dataset

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


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    ecoli_dataset = pd.read_csv("./data/ecoli.data", sep='  ',names=["sequence_names", "mcg","gvh","lip","chg","aac","alm1","alm2","decision"])
    #ecoli_dataset["decision"].replace(["cp","im","imU","imS","imL","om","omL","pp"], [0,1,2,3,4,5,6,7], inplace = True)
    #ecoli_dataset.replace('  ',',')

    #ecoli_dataset['mcg'] = ecoli_dataset['mcg'].astype(float)
    #ecoli_dataset['gvh'] = ecoli_dataset['gvh'].astype(float)
    #ecoli_dataset['lip'] = ecoli_dataset['lip'].astype(float)
    #ecoli_dataset['chg'] = ecoli_dataset['chg'].astype(float)
    #ecoli_dataset['aac'] = ecoli_dataset['aac'].astype(float)
    #ecoli_dataset['alm1'] = ecoli_dataset['alm1'].astype(float)
    #ecoli_dataset['alm2'] = ecoli_dataset['alm2'].astype(float)
    #ecoli_dataset['decision'] = ecoli_dataset['decision'].astype(float)

    #print(ecoli_dataset)
    k= 2
    clf = KNN(k)
    #iris = datasets.load_breast_cancer()
    X = ecoli_dataset.drop(["sequence names","decision"], axis = 1)
    y = ecoli_dataset.decision
    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))