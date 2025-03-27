"""
    Link:    https://github.com/codebasics/py/blob/master/ML/11_random_forest/Exercise/random_forest_exercise.ipynb
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris_dataset = load_iris()
df = pd.DataFrame(iris_dataset.data)
df['target'] = iris_dataset.target
x = df.drop('target',axis='columns')
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=40)
model.fit(x_train, y_train)

print(model.score(x_test, y_test))


