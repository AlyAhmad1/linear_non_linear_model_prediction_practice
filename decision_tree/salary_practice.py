import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv('salaries.csv')
input_d = df.drop(['salary_more_then_100k'], axis=1)
target = df['salary_more_then_100k']

company_encoder = LabelEncoder()
job_encoder = LabelEncoder()
degree_encoder = LabelEncoder()

input_d['company_n'] = company_encoder.fit_transform(df['company'])
input_d['job_n'] = job_encoder.fit_transform(df['job'])
input_d['degree_n'] = degree_encoder.fit_transform(df['degree'])

input_d = input_d.drop(['company', 'job', 'degree'], axis='columns')

dec_tree_model = tree.DecisionTreeClassifier()
dec_tree_model.fit(input_d, target)

# model score will be 1.0 because model tested and train on same data.
print(dec_tree_model.score(input_d, target))

print(dec_tree_model.predict([[0,1,0]]))

