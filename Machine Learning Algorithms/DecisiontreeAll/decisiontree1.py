import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.preprocessing import LabelEncoder


file = pd.read_csv('salaries.csv')
print(file.head())



inputs = file.drop('salary_more_then_100k',axis='columns')
target = file['salary_more_then_100k']

# Conversion string into numeric values using onehot Encoding Technique
l_company = LabelEncoder()
l_job = LabelEncoder()
l_degree = LabelEncoder()

inputs['company_n'] = l_company.fit_transform(inputs['company'])
inputs['job_n'] = l_company.fit_transform(inputs['job'])
inputs['degree_n'] = l_company.fit_transform(inputs['degree'])

print(inputs.head())

inputs_n = inputs.drop(['company','job','degree'],axis='columns')
print(inputs_n.head())

x_train,x_test,y_train,y_test =train_test_split(inputs_n,target,train_size=0.5)
print(len(x_train))

# Use of Decision tree classifier
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
model.predict(x_test)

print(model.score(x_test,y_test))






