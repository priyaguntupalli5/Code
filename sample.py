import math, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class KNN:
    def pre_processing(self, df):
        
        df["smoothness"].replace(["?"], ['0'], inplace = True)
        df['smoothness'] = df['smoothness'].astype(int)
        average = df["smoothness"].mean()
        df["smoothness"].replace(["?"], [average], inplace = True)

        df['diagnosis'] = df['diagnosis'].astype(int)
        df['radius'] = df['radius'].astype(int)
        df['texture'] = df['texture'].astype(int)
        df['perimeter'] = df['perimeter'].astype(int)
        df['area'] = df['area'].astype(int)
        df['smoothness'] = df['smoothness'].astype(int)
        df['compactness'] = df['compactness'].astype(int)
        df['concavity'] = df['concavity'].astype(int)
        df['concave points'] = df['concave points'].astype(int)
        df['decision'] = df['decision'].astype(int)
        return df
        
    def knn_implementation(self, x_train, y_train, x_test, K):
        dists, train_size = {}, len(x_train)
        for i in range(train_size):
            d = self.euclidian_dist(x_train.iloc[i], x_test)
            dists[i] = d
    
        k_neighbors = sorted(dists, key=dists.get)[:K]
    
        qty_label1, qty_label2 = 0, 0
        for index in k_neighbors:
            if int(y_train.iloc[index]) == 2:
                qty_label1 += 1
            else:
                qty_label2 += 1
            
        if qty_label1 > qty_label2:
            return 2
        else:
            return 4

    def euclidian_dist(self, x_train, x_test):
        dim, sum_ = len(x_train), 0
        for index in range(dim - 1):
            sum_ += math.pow(x_train[index] - x_test[index], 2)
        return math.sqrt(sum_)

    def mean_accuracy(self, accuracy):
        return sum(accuracy)/ len(accuracy)
        

if __name__ == "__main__":

    cancer_dataset = pd.read_csv("./data/breast-cancer-wisconsin.data", names=["id","diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","decision"])
    accuracy = []
    for i in range(10):
    # Shuffle the dataset 
        cancer_dataset = cancer_dataset.sample(frac=1)
        knn = KNN()
        df = knn.pre_processing(cancer_dataset)
        X = df.drop(["id","decision"], axis = 1)
        y = df.decision
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)
        kf = KFold(n_splits = 5)
        KFold(n_splits=2, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(X):
     #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        correct, K = 0, 15
        for i in range(0, len(X_test)):
            #print(x_train.iloc[i])
            label = knn.knn_implementation(X_train, y_train, X_test.iloc[i], K)
            if int(y_test.iloc[i]) == label:
                correct += 1
        accuracy.append(100 * correct / len(X_train))
    mean = knn.mean_accuracy(accuracy)
    print("Mean Accuracy of 10 folds: %.2f%%" % mean)
            
          