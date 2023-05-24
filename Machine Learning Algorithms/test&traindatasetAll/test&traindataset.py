import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Loading CSV Using Pandas Library:
CSV = pd.read_csv('train&test.csv.csv')
print(CSV.head())

# Plotting Graph Between Independent And Dependant Variables:
plot.figure()
plot.xlabel("Mileage")
plot.ylabel("Sell Price($)")
plot.scatter(CSV['Mileage'],CSV['Sell Price($)'])
plot.show()

plot.scatter(CSV['Age(yrs)'],CSV['Sell Price($)'])
plot.show()


# Defining X & Y as a Dependant & Indepedent Variable
x = CSV[['Mileage','Age(yrs)']]
y = CSV['Sell Price($)']

# Splitting Of Dataset in Training And Testing Dataset Using Sklearn(train_test_Model):

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

# Model is trained Using Linear Regression Algorithm

Model = LinearRegression()
Model.fit(x_train,y_train)
prediction = Model.predict(x_test)
print(prediction)
Accuracy = Model.score(x_test,y_test)
print("Accuracy of the Model is =",Accuracy)

# testing Example
prediction1= Model.predict([[46000,6]])
print(prediction1)


