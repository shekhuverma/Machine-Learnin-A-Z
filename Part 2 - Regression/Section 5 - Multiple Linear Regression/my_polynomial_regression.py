import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
#fitting linear reg to dataset
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)

#polynomial reg
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(4)
x_poly=poly_reg.fit_transform(x)
linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)
#plotting linear regression model
plt.scatter(x,y,color='red')
plt.plot(x,linear_reg.predict(x),color="blue")
plt.title("truth or bluff(liner)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

#plotting polynoial regression model
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,linear_reg2.predict(poly_reg.fit_transform(x_grid)),color="blue")
plt.title("truth or bluff(poly)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

#predicting a new result wiht linear regression
linear_reg.predict(6.5)
#predicting by polynomial regression model
linear_reg2.predict(poly_reg.fit_transform(6.5))



