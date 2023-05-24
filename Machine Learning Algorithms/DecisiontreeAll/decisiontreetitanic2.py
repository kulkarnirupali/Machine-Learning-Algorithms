import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.preprocessing import LabelEncoder


file = pd.read_csv('titanic.csv')
print(file.head())

inputs = file.drop('Survived',axis='columns')
target = file['Survived']
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
# Conversion string into numeric values using onehot Encoding Technique
l_Sex= LabelEncoder()

inputs['Sex_n'] = l_Sex.fit_transform(inputs['Sex'])
print(inputs.head())

inputs_n = inputs.drop(['PassengerId','Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
print(inputs_n.head())

x_train,x_test,y_train,y_test =train_test_split(inputs_n,target,test_size=0.2)
print(len(x_train))

# Use of Decision tree classifier
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
model.predict(x_test)

print("Accuracy of the model is ",model.score(x_test,y_test))






