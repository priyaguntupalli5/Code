import math, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class KNN:
    def pre_processing(self, df):
        ObjectColumns = df.select_dtypes(include=np.object).columns.tolist()
        df['lettr'] = [ord(item)-64 for item in df['lettr']]
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

    letter_dataset = pd.read_csv('./data/letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
    accuracy = []
    for i in range(10):
    # Shuffle the dataset 
        letter_dataset = letter_dataset.sample(frac=1)
        knn = KNN()
        df = knn.pre_processing(letter_dataset)
        X = df.drop(["lettr"], axis = 1)
        y = df.lettr
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
            
          