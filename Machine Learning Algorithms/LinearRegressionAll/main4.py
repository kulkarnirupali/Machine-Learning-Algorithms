# Linear Regression with multiple variables affection the result.....

# Linear Regression model with multiple variables as a independant variable

import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math

file = pd.read_csv("practice4.csv")
print(file.head())

file.experience = file.experience.fillna("zero")

file.experience = file.experience.apply(w2n.word_to_num)
print(file)

test_score = math.floor(file['test_score(out of 10)'].mean())
print(test_score)
file['test_score(out of 10)']= file['test_score(out of 10)'].fillna(test_score)
print(file)


Regression = linear_model.LinearRegression()
Regression.fit(file[['experience','test_score(out of 10)','interview_score(out of 10)']],file[['salary($)']])
Regression.predict(file[['experience','test_score(out of 10)','interview_score(out of 10)']])
print("Perfect")
print(Regression.predict([[12,10,10]]))
print("its done completely")
