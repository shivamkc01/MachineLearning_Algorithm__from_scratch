"""
Decision Tree implentation from scratch
This code you can use for learning purpose.

programmed by Shivam Chhetry
** 11-08-2021
"""

import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
"""
Calculating Entropy -> Entropy measure of purity in a node.
Range[0,1] 
0 - Best purity 
1 - worst purity
formula:-
    H(s) = -p(+)log(p+) - p(-)log(p(-))
    p+ = % of +ve class
    p- = % of -ve class

p(X) = #x/n
where, #x is no of occurrences 
        n  is no of total samples
E = - np.sum([p(X).log2(p(X))])
"""


def entropy(y):
    hist = np.bincount(y)  # this will calculate the number of occurrences of all class labels
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

"""
Let's create a helper class to store the information for our node.
we want to store 
1. The best split feature(feature)
2. The best split threshold
3. The left and the right child trees
4. If we are at a leaf node we also want to store the actual value , the most 
common class label
"""
class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    """
    Now we create a helper function to determine if we are at a leaf
    node
    """
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    # applying some  stopping criteria to stop growing
    # e.g: maximum depth, minimum samples at node, no more class distribution in node
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Applying stopping criteria
        if (
                depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value= leaf_value)
        # If we didn't need stopping criteria then we select the feature indices
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search : Loop over all features and over all thresholds(all possible feature values.
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thersh):
        """
        IG = E(parent) - [weighted average].E(childern)
        Example:
            S = [0,0,0,0,0,1,1,1,1,1], S1=[0,0,1,1,1,1,1], S2=[0,0,0]
            IG = E(S0) -[(7/10)*E(S1)+(3/10)*E(S2)]
            IG = 1 - [(7/10)*0.863+(3/10)*0] = 0.395

        Note: The higher the information gain that specific way of spliting decision tree will be taken up.
        """
        # parent E
        parent_entropy = entropy(y)
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thersh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weighted avg child E
        n = len(y)
        n_left_samples, n_right_samples = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_left_samples/n) * entropy_left + (n_right_samples/n) * entropy_right

        # return IG
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thersh):
        left_idxs = np.argwhere(X_column <= split_thersh).flatten()
        right_idxs = np.argwhere(X_column > split_thersh).flatten()
        return left_idxs, right_idxs

    def predict(self,X):
        # traverse tree
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        # counter will calculate all the no of occurrences of y
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0] # returns tuples, and we want only value so we again say index 0 [0]
        return most_common

if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    def accuracy(y_true, y_pred):
        acu = np.sum(y_true == y_pred)/len(y_pred)
        return acu

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy : ", acc)