import numpy as np
from collections import Counter

def entropy(labels):                                         
    label_counts = np.bincount(labels)                          # Count occurrences of each class using np.bincount
    pbs = label_counts / len(labels)                            # Calculate the probabilities/proportions
    entropy_value = -np.sum(pbs * np.log2(pbs + np.exp(-9)))    # Calculate the entropy using only non-zero probabilities to avoid log(0) # Adding small value to avoid log(0)
    return entropy_value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Internal node has feature, threshold, and children (left, right)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # Leaf node has value
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self,min_samples_split=2,max_depth=100,n_feats=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def calculate_information_gain(self, X_column, y, split_point): # Information Gain = Entropy (before split) – Weighted Average (Entropy after split)
        
        parent_entropy = entropy(y)
 
        left_idxs = np.argwhere(X_column <= split_point).flatten()   
        right_idxs = np.argwhere(X_column > split_point).flatten() 
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        l_y, r_y = y[left_idxs], y[right_idxs]
        e_l, e_r = entropy(l_y) , entropy(r_y)

        child_entropies = (len(l_y)/len(y)) * e_l + (len(r_y)/ len(y)) * e_r # weighted_avg_entropy of child nodes
        information_gain = parent_entropy - child_entropies
        
        return information_gain
    
    def _best_criteria(self,X,y,feat_idxs):
        best_gain = -1 # Information Gain = Entropy (before split) – Weighted Average (Entropy after split), so its safe to initialize it to a negative number rather than None
        for feat_idx in feat_idxs:
            X_relevant = X[:,feat_idx]
            split_points = np.unique(X_relevant)
            for split_point in split_points:

                information_gain = self.calculate_information_gain(X_relevant,y, split_point)
            
                if information_gain > best_gain:
                    best_gain = information_gain
                    split_idx = feat_idx
                    best_split_point = split_point
    
        return split_idx, best_split_point
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth                 # should not grow deeper than a specified depth
            or n_labels == 1                        # all samples at the current node belong to the same class
            or n_samples < self.min_samples_split   # the number of samples is too small
        ):
            leaf_value = self._most_common_label(y) # max pooling or assigning the maximum value as the class
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)
    
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        if len(counter) == 0:
            return None  # Or some default value for empty leaves
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _traverse_tree(self, x, node):
        # Stopping criteria, If the node is a leaf, return the value (class prediction) 
        if node.is_leaf_node():
            return node.value

        # Otherwise, continue traversing the tree based on the threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score

    X,y = datasets.load_breast_cancer()["data"], datasets.load_breast_cancer()["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    dt = DecisionTree(min_samples_split=10, max_depth=10)
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)
    
    print("Accuracy is", balanced_accuracy_score(y_test,y_pred))