# Logistic Regression with Binary classification of the feature
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



file = pd.read_csv('logisticregression1.csv.csv')
print(file.head())

plot.scatter(file.age,file.bought_insurance,marker='+',color='yellow')
plot.show()

x_train,x_test,y_train,y_test=train_test_split(file[['age']],file['bought_insurance'])
print(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)
model.predict(x_test)

score =model.score(x_test,y_test)
print(score)

Prediction = model.predict([[25]])
print(Prediction)

print("Predicted Result Successfully")