from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import numpy as np

folds= [2,3,5,10,15,30,50,75]


def loadIris(): 
    iris=datasets.load_iris() 
    irisX = iris.data 
                   
    irisY = iris.target 
                    
    return irisX,irisY

def makeKNN():
    knn = KNeighborsClassifier() 
    return knn

def makeBayes():
    gnb = GaussianNB()
    return gnb 

def doCrossVal(nfold,func,dataX,dataY):
   
    classifier=func()
    np.random.seed() 
   
   
    scores=cross_validation.cross_val_score(classifier,dataX,dataY,
                                            cv=nfold,scoring='accuracy')
    return scores

irisX,irisY=loadIris()
print "\n In these tests we are trying to create a model that will accurately predict future supervised data.  The problem of 'overfitting' can arise due to overtraining the test data. The KNN cross-validation method was the most accurate for the Iris dataset."
for n in folds:
           print "\nPerforming {}-fold cross-validation using a Bayesian calssifier:".format(n)
           scores=doCrossVal(n,makeBayes,irisX,irisY)
           print "The accuracies for each fold-test are: "+ ",".join(map(str,scores))
           print "The mean accuracy of the Bayesian classifier is {}".format(scores.mean())

           print "\nPerforming {}-fold cross-validation using a KNN calssifier:".format(n)
           scores=doCrossVal(n,makeKNN,irisX,irisY)
           print "The accuracies for each fold-test are: "+ ",".join(map(str,scores))
           print "The mean accuracy of the KNN classifier is {}".format(scores.mean())
