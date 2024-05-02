#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:24:42 2020

@author: thomas
@modified by: jing yang
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# the things you need to know if you want to use this method
# initialising a dtree object using tree.DecisionTreeClassifier()
# calling the .fit() method to train your dtree object
# using the .predict() method to produce the predictions of your dtree.
    
def Tut9_Q4_yeo():
    
    # load data
    iris_dataset = load_iris()
    
    # split dataset using train_test_split. Notice the arguments we past in
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                        iris_dataset['target'], 
                                                        test_size=0.20, 
                                                        random_state=0)
    
    # fit tree
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4) #so what is dtree?
    dtree = dtree.fit(X_train, y_train) 
    
    # predict
    y_trainpred = dtree.predict(X_train)
    y_testpred = dtree.predict(X_test)
    
    # print accuracies
    print("Training accuracy: ", metrics.accuracy_score(y_train, y_trainpred))
    print("Test accuracy: ", metrics.accuracy_score(y_test, y_testpred))    

    # Plot tree
    tree.plot_tree(dtree)
    plt.show()
    
Tut9_Q4_yeo()