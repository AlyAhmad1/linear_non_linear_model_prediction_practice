"""Exercise
------------------------------------------------------------------------------------------------
Use sklearn.datasets iris flower dataset to train your model using logistic regression. You need to figure out accuracy of your model and use that to predict different samples in your test dataset. In iris dataset there are 150 samples containing following features,
    Sepal Length
    Sepal Width
    Petal Length
    Petal Width
Using above 4 features you will clasify a flower in one of the three categories,
    Setosa
    Versicolour
    Virginica

Link: https://github.com/codebasics/py/blob/master/ML/8_logistic_reg_multiclass/8_logistic_regression_multiclass.ipynb
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
from matplotlib import pyplot as plt


iris_data_set = load_iris()

# view directories in dataset
print(dir(iris_data_set))

X_train, X_test, y_train, y_test = train_test_split(iris_data_set.data,iris_data_set.target,train_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(model.score(X_test,y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
