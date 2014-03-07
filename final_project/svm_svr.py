import pandas as pd
from numpy import nan
import numpy as np
import pylab as pl
from sklearn.preprocessing import StandardScaler
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold


df=pd.read_csv('wineries.csv')
from sklearn.svm import SVR

svr=SVR(kernel='rbf')
#svr=SVR(kernel='poly')
#svr=SVR(kernel='linear')



#target=map(float,df.Sales)
target=df.Sales.astype(float)
data=df[['Tweets','Likes','Checkins','Followers']].fillna(0.0)#.as_matrix()

#Plot raw data
fig,splots=pl.subplots(2,2)
for i,colname in enumerate(['Tweets','Likes','Checkins','Followers']):
    ax=splots[i%2,i/2]
    ax.scatter(data[colname],target)
    ax.set_xlabel(colname)
    ax.set_ylabel('Sales')


pl.show(block=False)




#cross-validation
nfolds=50
folds=KFold(len(data),n_folds=nfolds)
sales_predictions=[list() for x in range(len(data))]
rmse_arr=[]

for train_idx,test_idx in folds:
    #partition data into training/testing
    train_data=data.iloc[train_idx]
    train_target=target.iloc[train_idx].tolist()
    test_data=data.iloc[test_idx]
    test_target=target.iloc[test_idx].tolist()
    #scale data
    target_scaler=StandardScaler().fit(train_target)
    train_target_scaled=target_scaler.transform(train_target)
    test_target_scaled=target_scaler.transform(test_target)

    data_scaler=StandardScaler().fit(train_data)
    train_data_scaled=data_scaler.transform(train_data)
    test_data_scaled=data_scaler.transform(test_data)

    #fit and predict
    results=target_scaler.inverse_transform(svr.fit(train_data_scaled,train_target_scaled).predict(test_data_scaled))
    #save statistics

    for idx,prediction in zip(train_idx,results):
        sales_predictions[idx].append(prediction)

    rmse_arr.append(np.sqrt(mean_squared_error(results,test_target)))




#plot CV results
pl.figure()
pl.scatter(target,[np.mean(pred_arr) for pred_arr in sales_predictions]
    ,alpha=0.3)
pl.ylabel('Predicted Sales (average over {} folds)'.format(nfolds))
pl.xlabel('Sales')
pl.title("Mean RMSE over {} folds: {}".format(nfolds,np.mean(rmse_arr)))
pl.show()

