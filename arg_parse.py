import argparse
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import numpy as np

    
def loadIris(): #returns the data and labels for the iris dataset
    iris=datasets.load_iris() #load the needed dataset
    irisX = iris.data #the data is four columns per record - petal length/width
                   #and sepal length/width
    irisY = iris.target #this is the classification to iris types
                     #(Setosa, Versicolour, and Virginica)
    return irisX,irisY

def makeKNN():
    knn = KNeighborsClassifier() #Initialize a KNN classifier
    return knn

def makeBayes():
    gnb = GaussianNB() #Initialize a bayesian classifier
    return gnb 

def doCrossVal(nfold,func,dataX,dataY):
    #Performs cross validation for Bayes or KNN
    #given the number of folds to partition the input data into
    classifier=func()
    np.random.seed() #initialize random number generator
    #cross_val_score trains the appropriate model and performs cross validation
    #giving the accuracy score for each fold
    scores=cross_validation.cross_val_score(classifier,dataX,dataY,
                                            cv=nfold,scoring='accuracy')
    return scores
folds= [2,3,5,10,15,30,50,75]
irisX,irisY=loadIris()
parser = argparse.ArgumentParser()
parser.add_argument("knn",nargs="*")
parser.add_argument("nb",nargs="*")

args = parser.parse_args()
if args.knn:
     for n in folds:
          print "\nPerforming {}-fold cross-validation using a KNN calssifier:".format(n)
          scores=doCrossVal(n,makeKNN,irisX,irisY)
          print "The accuracies for each fold-test are: "+ ",".join(map(str,scores))
          print "The mean accuracy of the KNN classifier is {}".format(scores.mean())
elif args.nb:
	  for n in folds:
	  	  print "\nPerforming {}-fold cross-validation using a Bayesian calssifier:".format(n)
          scores=doCrossVal(n,makeBayes,irisX,irisY)
          print "The accuracies for each fold-test are: "+ ",".join(map(str,scores))
          print "The mean accuracy of the Bayesian classifier is {}".format(scores.mean())

