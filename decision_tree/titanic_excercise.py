"""
CSV file is available to download at https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/titanic.csv

In this file using following columns build a model to predict if person would survive or not,
Pclass
Sex
Age
Fare
Calculate score of your model

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('titanic.csv')


# print(df_new.head())
lb_sex = LabelEncoder()
df['Sex_enc'] = lb_sex.fit_transform(df['Sex'])
input_d = df.drop(['Survived', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis='columns')
target = df['Survived']

train_x, test_x, train_y, test_y = train_test_split(input_d, target, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(train_x, train_y)

print(model.score(test_x, test_y))
