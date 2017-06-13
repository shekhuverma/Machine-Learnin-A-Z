
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
#importing dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# encoding the categorisal data
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
#avoiding dummy variable trap
x=x[:,1:]

#splitiing the dataset and test test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#fitting multiple liner regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#building model by backward elimination
import statsmodels.formula.api as sm
x=np.append(np.ones((50,1),dtype='int8'),x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(y,x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(y,x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(y,x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(y,x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(y,x_opt).fit()
regressor_OLS.summary()


