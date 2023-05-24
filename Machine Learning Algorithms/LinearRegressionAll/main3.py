# Linear Regression model with multiple variables as a independant variable


import pandas as pd
import math
import matplotlib.pyplot as plot
from sklearn import linear_model


file = pd.read_csv("practice3.csv")
print(file.head())

mediun =math.floor(file.bedrooms.median())
print(mediun)

file.bedrooms = file.bedrooms.fillna(mediun)
print(file)

regressionmodel = linear_model.LinearRegression()
regressionmodel.fit(file[['area','bedrooms','age']],file.price)
regressionmodel.predict(file[['area','bedrooms','age']])
prediction = regressionmodel.predict([[3000,3,40]])
print(prediction)


print(regressionmodel.coef_)
print(regressionmodel.intercept_)


plot.figure()
plot.xlabel("independant variables")
plot.ylabel("dependant variable")
plot.scatter(file.area,file.price)
plot.grid()
plot.show()
