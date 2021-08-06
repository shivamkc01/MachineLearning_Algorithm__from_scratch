"""
K-Nearest_Neighbour algorithm pytorch from scratch 
where you can either use 2-loops (inefficient), 1-loop (better)
or a heavily vectorized zero-loop implementation.

Programmed by Shivam Chhetry
* 06-08-2021
"""

import numpy as np

class KNearestNeighbour:
    def __init__(self, k):
        self.k = k   #
        self.eps = 1e-6  ## this is i'm using to make my calculation to be simple by adding small(chhotu) number
    
    def train(self, X, y):
        self.X_train = X
        self.Y_train = y
    
    def predict(self, X_test, num_loops=2):
        if num_loops == 0:
            distances = self.compute_distance_victorized(X_test)
        elif num_loops == 1:
            distances = self.compute_distance_one_loop(X_test)
        else:
            distances = self.compute_distance_two_loops(X_test)

            
            
        return self.predict_labels(distances)
    
    
    def compute_distance_two_loops(self, X_test):
        #Naive, inefficient way
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            for j in range(num_train):
                distances[i,j] = np.sqrt(self.eps + np.sum((X_test[i, :] - self.X_train[j, :])**2))
        
        return distances       
    
    def compute_distance_one_loop(self, X_test):
        """
        Much better than two-loops but not as fast as fully vectorized version.

        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            distances[i,:] = np.sqrt(self.eps + np.sum((self.X_train - X_test[i,:])**2, axis=1))
        
        return distances
    
    def compute_distance_victorized(self, X_test):
        
        
        """
        Idea: if we have two vectors a, b (two examples)
        and for vectors we can compute (a-b)^2 = a^2 - 2a (dot) b + b^2
        expanding on this and doing so for every vector lends to the 
        heavy vectorized formula for all examples at the same time.
        """
        
        
        #(X_test-X_train)^2 = (X_test^2 - 2*X_test*X_train + X_train^2)
        
        X_train_squared = np.sum(X_test**2, axis=1, keepdims=True)
        X_test_squared = np.sum(self.X_train**2, axis=1, keepdims=True)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)
        
        # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)

        return np.sqrt(
            self.eps + X_test_squared -2*two_X_test_X_train + X_train_squared.T
            )
        
        
        
    
    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_closest_classes = self.Y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))
        
        return y_pred
    
    
if __name__ == "__main__":
    
    """
    (X_test-X_train)^2 = (X_test^2 - 2*X_test*X_train + X_train^2)
    train = np.random.randn(1,4)
    test = np.random.randn(1,4)
    num_examples = train.shape[0]
    
    KNN.train(train,np.zeros(num_examples))
    
    distances = np.sqrt(np.sum(test**2, axis=1, keepdims=True) + np.sum(train**2, axis=1, keepdims=True) - 2*np.sum(test*train)) ##keepdims returns (1,1) instead of (1,)
    
    corr_distance = KNN.compute_distance_two_loops(test)
        
    #print(f'The difference is : {np.sqrt(np.sum((corr_distance - distances)**2))}')


    """
    X = np.loadtxt("https://raw.githubusercontent.com/aladdinpersson/Machine-Learning-Collection/master/ML/algorithms/knn/example_data/data.txt", delimiter=',')
    y = np.loadtxt("https://raw.githubusercontent.com/aladdinpersson/Machine-Learning-Collection/master/ML/algorithms/knn/example_data/targets.txt")
    
    KNN = KNearestNeighbour(k=2)
    KNN.train(X, y)
    y_pred = KNN.predict(X, num_loops=1)

    
    
    print(f'accuracy:  {sum(y_pred==y)/y.shape[0]}')
    










