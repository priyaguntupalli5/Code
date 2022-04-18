import math, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class KNN:
    def pre_processing(self, df):
        
        df["decision"].replace(["cp","im","imU","imS","imL","om","omL","pp"], [0,1,2,3,4,5,6,7], inplace = True)

        df['mcg'] = df['mcg'].astype(float)
        df['gvh'] = df['gvh'].astype(float)
        df['lip'] = df['lip'].astype(float)
        df['chg'] = df['chg'].astype(float)
        df['aac'] = df['aac'].astype(float)
        df['alm1'] = df['alm1'].astype(float)
        df['alm2'] = df['alm2'].astype(float)
        df['decision'] = df['decision'].astype(float)
        return df
        
    def knn_implementation(self, x_train, y_train, x_test, K):
        dists, train_size = {}, len(x_train)
        for i in range(train_size):
            d = self.euclidian_dist(x_train.iloc[i], x_test)
            dists[i] = d
    
        k_neighbors = sorted(dists, key=dists.get)[:K]
    
        qty_labels=[0,0,0,0,0,0,0,0]
        for index in k_neighbors:
            if y_train.iloc[index] == 0:
                qty_labels[0] += 1
            elif y_train.iloc[index] == 1:
                qty_labels[1] += 1
            elif y_train.iloc[index] == 2:
                qty_labels[2] += 1
            elif y_train.iloc[index] == 3:
                qty_labels[3] += 1
            elif y_train.iloc[index] == 4:
                qty_labels[4] += 1
            elif y_train.iloc[index] == 5:
                qty_labels[5] += 1
            elif y_train.iloc[index] == 6:
                qty_labels[6] += 1
            elif y_train.iloc[index] == 7:
                qty_labels[7] += 1     
        max_label =  max(qty_labels)
        return qty_labels.index(max_label)

    def euclidian_dist(self, x_train, x_test):
        dim, sum_ = len(x_train), 0
        for index in range(dim - 1):
            sum_ += math.pow(x_train[index] - x_test[index], 2)
        return math.sqrt(sum_)

    def mean_accuracy(self, accuracy):
        return sum(accuracy)/ len(accuracy)
        

if __name__ == "__main__":

    ecoli_dataset = pd.read_csv("./data/ecoli.data", names=["sequence names", "mcg","gvh","lip","chg","aac","alm1","alm2","decision"])
    accuracy = []
    for i in range(10):
    # Shuffle the dataset 
        ecoli_dataset = ecoli_dataset.sample(frac=1)
        knn = KNN()
        df = knn.pre_processing(ecoli_dataset)
        X = df.drop(["sequence names","decision"], axis = 1)
        y = df.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #kfdata = KFold(n_splits = 5)
        correct, K = 0, 15
        for i in range(0, len(X_test)):
            #print(x_train.iloc[i])
            label = knn.knn_implementation(X_train, y_train, X_test.iloc[i], K)
            if y_test.iloc[i] == label:
                correct += 1
        accuracy.append(100 * correct / len(X_train))
    mean = knn.mean_accuracy(accuracy)
    print("Mean Accuracy of 10 folds: %.2f%%" % mean)
            
          