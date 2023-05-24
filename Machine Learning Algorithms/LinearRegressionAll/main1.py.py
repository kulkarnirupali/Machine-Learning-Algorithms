# Linear Regression model with single variables as a independant variable


import pandas as pd
import matplotlib.pyplot as plot
from sklearn import linear_model



Data = pd.read_csv("practice1.csv")

plot.figure
plot.xlabel("Area")
plot.ylabel("Prices")
plot.scatter(Data.AREA,Data.PIECES,color='red',marker='.')
plot.show()
regression = linear_model.LinearRegression()
regression.fit(Data[['AREA']],Data[['PIECES']])
regression.predict(Data[['AREA']])
print(regression.predict([[10000]]))
print(regression.coef_)
print(regression.intercept_)



# Use of Pickle Model for saving model for future use :

import pickle

with open ('dataframe.txt','wb') as file:
    pickle.dump(regression,file)

with open('dataframe.txt','rb') as file:
    Dataframe = pickle.load(file)
    result = Dataframe.predict([[3000]])
    print(result)