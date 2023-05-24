# Linear Regression model with single variables as a independant variable

import pandas as pd
import matplotlib.pyplot as plot
from sklearn import linear_model


file = pd.read_csv("practice2.csv")
print(file.head())


plot.figure
plot.xlabel("year")
plot.ylabel("pci")

plot.scatter(file.year,file.pci)
plot.grid()
plot.show()



regression = linear_model.LinearRegression()
regression.fit(file[['year']],file[['pci']])
regression.predict(file[['year']])
print(regression.predict([[2020]]))
print("Its done perfectly")
