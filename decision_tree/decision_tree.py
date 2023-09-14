import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self, max_depth=None):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.max_depth = 5
        self.tree = None
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        def _fit_tree(X, y, depth):
            # Check if there are any attributes left to split on
            if self.max_depth is None or y.nunique() == 1 or len(X.columns) == 0:
                return y.mode()[0]
            
            # Find the attribute with the highest information gain
            best_attr = information_gain(X, y)
            
            #Check if the information gain is 0
            if best_attr is None:
                return y.mode()[0]
            
            # Define the tree
            tree = {best_attr:{}}
            for val in np.unique(X[best_attr]):
                # Create a subset of the data for each unique value in the best attribute
                X_subset = X[X[best_attr] == val].drop(columns=[best_attr])
                y_subset = y[X[best_attr] == val]
                # Recursively call the function to create the branches of the tree
                tree[best_attr][val] = _fit_tree(X_subset, y_subset, depth+1)
            
            return tree

        self.tree = _fit_tree(X, y, depth=0)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        rules = self.get_rules()
        outcomes = pd.Series([rule[-1] for rule in rules])
        y_pred = []
        for i in range(len(X)):
            for rule in rules:
                if all(item in X.iloc[i].items() for item in rule[0]):
                    y_pred.append(rule[-1])
                    break
                # If no rule is found, predict the most common label
                elif rule == rules[-1]:
                    y_pred.append(outcomes.mode()[0])
                
        return np.array(y_pred)
            
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        def _get_rules(tree, rules=[], rule=[]):
            for key, value in tree.items():
                if isinstance(value, dict):
                    rule.append(key)
                    _get_rules(value, rules, rule)
                    rule.pop()
                else:
                    rule.append(key)
                    rules.append((rule[:], value))
                    rule.pop()

            def rewrite_rules(rules):
                new_rules = []
                for rule in rules:
                    new_rule = []
                    for i in range(int(len(rule[0])/2)):
                        new_rule.append((rule[0][2*i], rule[0][2*i+1]))
                    new_rule = (new_rule, rule[1])
                    new_rules.append(new_rule)
                return new_rules
                
            return rewrite_rules(rules)
        return _get_rules(self.tree)

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    #print(y_true.shape, y_pred.shape)
    
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



def information_gain(X, y):
    """
    Computes the information gain of splitting on each attribute

    Args:
        X (pd.DataFrame): a matrix with discrete value where
            each row is a sample and the columns correspond
            to the features.
        y (pd.Series): a vector of discrete ground-truth labels

    Returns:
        The name of the attribute with the highest information gain
    """
    
    # Check if there are any attributes left to split on
    if len(X.columns) == 0:
        return None

    # Create S, the set of unique labels and their counts
    uniq = np.unique(y)
    S = np.zeros(len(uniq))
    for i in range(len(uniq)):
        S[i] = np.sum(y == uniq[i])

    # Calculate the entropy of S
    entropy_S = entropy(S)

    # Create a dictionary to store the information gain for each attribute
    info_gain = {key:None for key in X.columns}

    # Calculate the information gain for each attribute
    for i in range(len(X.columns)):
        # Create a list of unique values for the attribute
        attr_uniq = np.unique(X.iloc[:,i])
        # Create a list of counts for each unique value
        counts = np.zeros((len(attr_uniq),len(uniq)))
        # Iterate through each unique value and count instances of each label
        for j in range(len(counts)):
            for k in range(len(uniq)):
                counts[j,k] = np.sum((X.iloc[:,i] == attr_uniq[j]) & (y == uniq[k])) 
        info_gain[X.columns[i]] = entropy_S - np.sum([np.sum(counts[j,:])/np.sum(S)*entropy(counts[j,:]) for j in range(len(counts))])

    # Select the attribute with the highest information gain
    best_attr = max(info_gain, key=info_gain.get)
    # Return the attribute with the highest information gain
    return best_attr




