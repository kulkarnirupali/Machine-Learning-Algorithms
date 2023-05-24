import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

flower = load_iris()
print(dir(flower))

x_train,x_test,y_train,y_test = train_test_split(flower.data,flower.target,train_size=0.8)
print(len(x_train))
print(len(x_test))

Model = LogisticRegression()
Model.fit(x_train,y_train)
Model.predict(x_test)
Accuracy = Model.score(x_test,y_test)
print("Accuracy of the Model is ",Accuracy)

y_predicted = Model.predict(x_test)
cm = confusion_matrix(y_test,y_predicted)
print("Confusion matrix can be drawn as below")

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted Result")
plt.ylabel("True Result")
plt.show()
print("Successfully Completed")

