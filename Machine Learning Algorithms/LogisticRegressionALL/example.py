import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

# Loading the dataset from sklearn Library
digits = load_digits()
print(dir(digits))



plt.matshow(digits.images[1])
plt.gray()
plt.show()

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,train_size=0.8)
print(len(x_train))
print(len(x_test))

Model = LogisticRegression()
Model.fit(x_train,y_train)
Model.predict(x_test)
print("Model trained successfully")

# Checking Accuracy by Score method
print(Model.score(x_test,y_test))

# Testing random example
print(Model.predict(digits.data[[67]]))

# Use of Confusion matrix
y_predicted = Model.predict(x_test)
cm = confusion_matrix(y_test,y_predicted)

# Plotting confusion matrix using Seaborn Library
plt.show()
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print("finely done")
