#  THis Python Script is About One Hot Encoding i.e. Dummy Encoding


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot


df = pd.read_csv("carprices.csv")
print(df)

dummy = pd.get_dummies(df[['CarModel']])
print(dummy)

merged = pd.concat([df,dummy],axis='columns')
print(merged)

final = merged.drop(['CarModel','CarModel_Audi A5'],axis ='columns')
print(final)

model = LinearRegression()
x = final.drop('Price',axis='columns')
y = final.Price

model.fit(x,y)
prediction = model.predict([[87000,7,0,1]])
prediction1 = model.predict([[45000,4,0,0]])
print(prediction,prediction1)
print(model.score(x,y))


# Use of Matplotlib library

plot.figure
plot.xlabel("Age(yrs)")
plot.ylabel("Price")
plot.scatter(df.Age,df.Price,color='skyblue',marker='.')
plot.show()


