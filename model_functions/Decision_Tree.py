import numpy as np
import pandas as pd

class Decision_Tree:
  def __init__(self, max_depth=6, min_samples_split=1000):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

  def check_stop(self, y, depth):  #checking function to see when tree shouldn't expand anymore
        if depth >= self.max_depth:
            return True
        elif len(y) <= self.min_samples_split:
            return True
        return False
  
  def find_best_split(self,X,y):                #Finding the best feature and threshold for a split 
    best_feat , best_tresh, best_rss = None, None, float('inf')
    for feat in range(X.shape[1]):
        for t in np.unique(X[:,feat]):
            left= y[X[:,feat]<=t]
            right=y[X[:,feat]>t]
            if len(left) ==0 or len(right) ==0:
                continue
            rss= (len(left) * np.var(left) +len(right) * np.var(right)) / len(y)
            if rss < best_rss:
                best_feat,best_tresh,best_rss =feat,t,rss
    return best_feat,best_tresh

  def build_tree(self,X,y,depth=0):          #buidling the tree
    if self.check_stop(y, depth):
       return np.mean(y)
    feat,thresh = self.find_best_split(X,y)
    left_mask=X[:,feat] <= thresh
    right_mask= ~left_mask
    left_branch= self.build_tree(X[left_mask],y[left_mask],depth+1)
    right_branch=self.build_tree(X[right_mask], y[right_mask],depth+1)
    return(feat,thresh,left_branch,right_branch)
  
  def fit(self, X, y):
    self.tree = self.build_tree(X, y)

  def print_tree(self,node, feature_names, depth=0):
    indent = "  " * depth
    if not isinstance(node, tuple):
        print(f"{indent}Predict Risk = {node:.3f}")
        return
    feat, thresh, left, right = node
    print(f"{indent}if {feature_names[feat]} <= {thresh}:")
    self.print_tree(left, feature_names, depth+1)
    print(f"{indent}else:")
    self.print_tree(right, feature_names, depth+1)
  
