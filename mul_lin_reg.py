#APPLYING MLR FOR Hospital infection data....

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly as pl

data = pd.read_csv("hospital_infct.txt",sep='\t')
print( data.head() )
print( data.columns )

#get the target...
y=data.InfctRsk.values #values convert values onto numpy array

x_data = data.drop(["InfctRsk","ID"], axis=1) #except for target the other columns is our x data

#First of all, we need to make normalization on data. 
#X_normalized = (x - x minimum)/(x maximum - x minimum)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

#Convert the data frame to matrix...
X=x_train.to_numpy()
print(X.shape)
v=np.ones(( len(x_train),1))
X=np.append(v, X,axis=1)
print(X.shape)

temp1=  np.dot (np.transpose(X), X )

t1=np.linalg.pinv( temp1)  #Compute pseudo inverse...

t2 = np.dot ( np.transpose(X),y_train)

beta= np.dot (t1,t2)
print(beta)

#prediction.....
x_test=x_test.to_numpy()  #Append 1....
v2=np.ones(( len(x_test),1))
x_test=np.append(v2, x_test,axis=1)
print(x_test.shape,beta.shape)

pred= np.dot(x_test,beta)
print(pred)

RSS= np.sum( (y_test - pred)* (y_test - pred) )
print("RSS is ", RSS)






