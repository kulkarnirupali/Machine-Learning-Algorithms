# Logistic Regression with Binary classification of the feature
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



file = pd.read_csv('logisticregression2.csv.csv')
print(file.head())


# bar type of graph
pd.crosstab(file.salary,file.left).plot(kind='bar')
plot.show()

pd.crosstab(file.Department,file.left).plot(kind='bar')
plot.show()

# one hot encoding technique to convert string into integer
dummy = pd.get_dummies(file[['Department']],prefix='salary')

# merging of dummy and original dataset
merged = pd.concat([file,dummy],axis='columns')
print(merged.head(3))

# Drop encoded value i.e String Part
merged.drop('salary',axis='columns',inplace=True)
merged.drop('Department',axis='columns',inplace=True)
print(merged.head(3))
print("done")

x=merged
print(x.head(3))

y=file.left
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)
model.predict(x_test)

score =model.score(x_test,y_test)
print(score)

print("Predicted Result Successfully")