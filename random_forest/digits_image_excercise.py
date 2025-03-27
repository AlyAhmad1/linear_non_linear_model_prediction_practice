"""
    Link:    https://github.com/codebasics/py/blob/master/ML/11_random_forest/11_random_forest.ipynb
"""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


digits = load_digits()

# check dir-structure of digits dataset.
print(dir(digits))

df = pd.DataFrame(digits.data)
df['target'] = digits.target

X = df.drop('target',axis='columns')
y = df.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=20)
model.fit(x_train, y_train)

print(model.score(x_test, y_test))


y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
