#!/usr/bin/python

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import numpy as np

import sys

try:
    nfolds=int(sys.argv[1]) 
                           
except:
    print "First argument must be the number of folds for cross-validation"
    exit()



def do_logreg():
    from sklearn.preprocessing import Binarizer, scale
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,classification_report
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_val_score
    from sklearn.grid_search import GridSearchCV
    from scipy.stats import expon
    import pandas
    ### load data
    col_names=['mpg','cylinders','displacement','horsepower','weight',
               'acceleration','model_year','origin','car_name']
    df=pandas.read_csv('auto_mpg.csv')
    df.columns=col_names
    df=df.drop('car_name',1)
    
    lr=LogisticRegression()
    bn=Binarizer(threshold=df['mpg'].mean())
    print "Performing binarization of the mpg variable into above/below average classes"
    target=bn.fit_transform(df['mpg'])
    data=df.drop('mpg',1)
    data=scale(data)
    print "Splitting into training and test sets"
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5,random_state=0)

    grid=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
    print 'Searching for optimal C in {} using {}-fold validation on test set '.format(grid,nfolds)
    tuned_parameters=[{'C':grid}]
    clf=GridSearchCV(lr,tuned_parameters,cv=nfolds,scoring='accuracy')
    clf.fit(data_train,target_train)
    for params, mean_score,_ in clf.grid_scores_:
        print "{}: Mean accuracy {}".format(params,mean_score)

    
    print  """Cross-validating above/below average mpg prediction
        using {}-fold validation on the test dataset.
        Using the best estimator: {}
        """.format(nfolds,clf.best_estimator_)
        
    mean_cross=np.mean(cross_val_score(clf.best_estimator_,data_test,target_test,cv=nfolds))

    print "Mean cross-validated accuracy after optimization is: {}".format(mean_cross)

    
    


do_logreg()
