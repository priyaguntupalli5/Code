import math, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class KNN:
    def pre_processing(self, df):
        
        df.replace(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],inplace = True)
        df["decision"] = df['decision'].astype(int)
        df["cap-shape"] = df['cap-shape'].astype(int)
        df["cap-surface"] = df['cap-surface'].astype(int)
        df["cap-color"] = df['cap-color'].astype(int)
        df["bruises"] = df['bruises'].astype(int)
        df["odor"] = df['odor'].astype(int)
        df["gill-attachment"] = df['gill-attachment'].astype(int)
        df["gill-spacing"] = df['gill-spacing'].astype(int)
        df["gill-size"] = df['gill-size'].astype(int)
        df["gill-color"] = df['gill-color'].astype(int)
        df["stalk-shape"] = df['stalk-shape'].astype(int)
        df["stalk-root"] = df['stalk-root'].astype(int)
        df["stalk-surface-above-ring"] = df['stalk-surface-above-ring'].astype(int)
        df["stalk-surface-below-ring"] = df['stalk-surface-below-ring'].astype(int)
        df["stalk-color-above-ring"] = df['stalk-color-above-ring'].astype(int)
        df["stalk-color-below-ring"] = df['stalk-color-below-ring'].astype(int)
        df["veil-type"] = df['veil-type'].astype(int)
        df["veil-color"] = df['veil-color'].astype(int)
        df["ring-number"] = df['ring-number'].astype(int)
        df["ring-type"] = df['ring-type'].astype(int)
        df["spore-print-color"] = df['spore-print-color'].astype(int)
        df["population"] = df['population'].astype(int)
        df["habitat"] = df['habitat'].astype(int)
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

    mushroom_dataset = pd.read_csv('mushroom.df', names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment", 
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", 
    "population", "habitat"])
    accuracy = []
    for i in range(10):
    # Shuffle the dataset 
        mushroom_dataset = mushroom_dataset.sample(frac=1)
        knn = KNN()
        df = knn.pre_processing(mushroom_dataset)
        X = df.drop(["decision"], axis = 1)
        y = df.decision
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #kfdata = KFold(n_splits = 5)
        correct, K = 0, 15
        for i in range(0, len(X_test)):
            #print(x_train.iloc[i])
            label = knn.knn_implementation(X_train, y_train, X_test.iloc[i], K)
            if int(y_test.iloc[i]) == label:
                correct += 1
        accuracy.append(100 * correct / len(X_train))
    mean = knn.mean_accuracy(accuracy)
    print("Mean Accuracy of 10 folds: %.2f%%" % mean)
            
          