from sklearn import svm
from sklearn import cross_validation
from sklearn import linear_model
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import time

def Pyplot(X,Y):
	temp_media = [] 
	for i in range(0,len(X)):
	#	print X[i]
  		temp = X[i]	
		temp = temp[1:len(temp)]
		temp_tot = 0
		for j in range(len(temp)):
			temp_tot = temp_tot + temp[j]
		lst = []
		lst.append(X[i][0])
		lst.append(temp_tot/len(temp))
		temp_media.append(lst)

	x_min,x_max,y_min,y_max = 10000,0,10000,0
	for i in range(len(temp_media)):
		temp = temp_media[i]
		x_min = min(temp[0],x_min)
		x_max = max(x_max,temp[0])

		y_min = min(temp[1],y_min)
		y_max = max(temp[1],y_max)

	a,b = 1,30
	for i in range(len(temp_media)):
		temp = temp_media[i]
		temp[0] = ((b - a) * (temp[0] - x_min) * 1.0)/((x_max - x_min) + a)
		temp[1] = ((b - a) * (temp[1] - y_min) * 1.0)/((y_max - y_min) + a)
		temp_media[i] = temp

#	print temp_media

	x_min,x_max,y_min,y_max = 10000,0,10000,0
	for i in range(len(temp_media)):
		temp = temp_media[i]
		x_min = min(temp[0],x_min)
		x_max = max(x_max,temp[0])

		y_min = min(temp[1],y_min)
		y_max = max(temp[1],y_max)

#	print x_min,x_max,y_min,y_max

	C = 1.0
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(temp_media, Y)
	lin_svc = svm.LinearSVC(C=C).fit(temp_media, Y)
	h = .02
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#	print temp_media
	titles = ['SVM with RBF kernel',"SVM using linearSVC"]
	for i, clf in enumerate((rbf_svc,lin_svc)):
		pl.subplot(2, 2, i + 1)
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
		pl.axis('off')
		for j in range(len(temp_media)):
#			print temp_media[j]
			pl.scatter(temp_media[j][0], temp_media[j][1], c=Y[j], cmap=pl.cm.Paired)
		pl.title(titles[i])
	pl.show()

def ROC_CURVE(X,Y):
	half = int(len(X) / 2)
	X_train, X_test = X[:half], X[half:]
	y_train, y_test = Y[:half], Y[half:]
#	print len(X_train),len(y_train)
#	print len(X_test),len(y_test)
	classifier = svm.SVC(kernel = "rbf" , probability=True)
	classifier.fit(X_train, y_train)
	probas_ = classifier.predict_proba(X_test)
#	print probas_
	fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)
	print("Area under the ROC curve for winery data set: %f" % roc_auc)
	pl.clf()
	pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('sales')
	pl.ylabel('social media data')
	pl.title('Receiver operating characteristic example for winery data set')
	pl.legend(loc="lower right")
	pl.show()

def logistic_regression(X,Y):
	num_folds = 10
	subset_size = len(X)/num_folds

	for i in range(num_folds):
		test_dt = X[i*subset_size:][:subset_size]
		train_dt = X[:i*subset_size] + X[(i+1)*subset_size:]
		test_label = Y[i*subset_size:][:subset_size]
		train_label = Y[:i*subset_size] + Y[(i+1)*subset_size:]
		clf = linear_model.LogisticRegression()
		clf.fit(train_dt,train_label)
		scores = clf.predict_proba(test_dt)
		error = 0.0
		for j in range(0,len(scores)):
			temp = scores[j]
			if temp[0] > temp[1]:
				clss = 0
			else:
				clss = 1
			if clss == 0:
				error = error + abs(test_label[j] - temp[0]) ** 2
			else:
				error = error + abs(temp[1] - test_label[j]) ** 2
		print "RMSE For %d fold using Logistic Regression in winery set: %f" %(i+1,error)


def svm_classify(X,Y):
	num_folds = 10
	subset_size = len(X)/num_folds

	for i in range(num_folds):
		test_dt = X[i*subset_size:][:subset_size]
		train_dt = X[:i*subset_size] + X[(i+1)*subset_size:]
		test_label = Y[i*subset_size:][:subset_size]
		train_label = Y[:i*subset_size] + Y[(i+1)*subset_size:]
		clf = svm.SVC(probability = True)
		clf.fit(train_dt,train_label)
		scores = clf.predict_proba(test_dt)
		error = 0.0
		for j in range(0,len(scores)):
			temp = scores[j]
			if temp[0] > temp[1]:
				clss = 0
			else:
				clss = 1
			if clss == 0:
				error = error + abs(test_label[j] - temp[0]) ** 2
			else:
				error = error + abs(temp[1] - test_label[j]) ** 2
		print "RMSE For %d fold using SVM suppor vector classification in winery set: %f" %(i+1,error)


def main():
	fname = open("wineries.csv","r")
	line = fname.readline()
	X = []
	Y = []
	for line in fname.readlines():
		line = line.strip()
		lst = line.split(',')
		for i in range(2,len(lst)):
			if "na" in lst[i] or lst[i] == "0" or not lst[i] or lst[i] == "N/A":
				lst[i] = 1
			else:
				lst[i] = int(lst[i])
		temp_x = lst[2:len(lst)]
#		print temp_x
		X.append(temp_x)
	
	total_sales = 0
	total_likes = 0
	for i in range(0,len(X)):
		temp = X[i]
		total_sales = total_sales + temp[0]

	avg_sales = (total_sales * 1.0) / len(X)

	avg_likes = []
	for i in range(0,len(X)):
		temp = X[i]
		temp = temp[1:len(temp)]
		total_cnt = 0
		for j in range(0,len(temp)):
			total_cnt = total_cnt + temp[j]
		avg_likes.append((total_cnt * 1.0) / len(temp))

	total_avg_sales = 0
	for i in range(0,len(avg_likes)):
		total_avg_sales = total_avg_sales + avg_likes[i]

	total_avg_media = total_avg_sales * 1.0 / len(avg_likes)
#	print avg_sales,total_avg_media

	for i in range(0,len(X)):
		flag = False
		flag2 = False
		main_flag = False
		temp = X[i]
		if temp[0] >= avg_sales:
			flag = True

		temp_tot = 0
		for j in range(1,len(temp)):
			temp_tot = temp_tot + temp[j]
		temp_avg = (temp_tot * 1.0)/(len(temp)-1)

		if temp_avg >= total_avg_media:
			flag2 = True

		if flag == True and flag2 == True:
			main_flag = True

		if flag == False and flag2 == False:
			main_flag = True

		if main_flag:
			Y.append(1)
		else:
			Y.append(0)
#	print Y
	return X,Y

if __name__ == "__main__":
	X,Y = main()
	print "CROSS VALIDATION USING SVM CLASSIFICATION:\n"
	svm_classify(X,Y)
	print "\n\n\n"
	print "CROSS VALIDATION USING LOGISTIC REGRESSION:\n"
	logistic_regression(X,Y)
	print "\n\n\n"
	print "Select Any one of two:\
		   \n1)For ROC Cruve press 1\
		   \n2) For pyplot press 2"

	choice = raw_input()
	choice = int(choice)
	if choice == 1:
		print "\n\n\n"
		print "ROC CURVE FOR WINERY DATA SET:\n"
		ROC_CURVE(X,Y)
	elif choice == 2:
		print "\n\n\n"
		print "Plotting data of wineries on graph using pyplot:\n"
		Pyplot(X,Y)
	else:
		print "Wrong choice\n"
	print "Exiting the program\n"

