import pandas as pd
from word2number import w2n
import math
from sklearn import linear_model

'''
linear regression eq: y = mx + b
y: dependent variable
x: independent variable
m: coefficient, formula: (sum of product of deviation)/sum of square of deviation for x 
b: intercept, formula: Mean of Y - (m * Mean of x)
'''

df = pd.read_csv('hiring.csv')
df['experience'] = df['experience'].fillna('Zero')

df['experience'] = df['experience'].apply(w2n.word_to_num)

# df['test_score(out of 10)'].median()
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(math.floor(df['test_score(out of 10)'].mean()))
print(df)

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)','interview_score(out of 10)']], df['salary($)'])

print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))
# # print(reg.coef_)
# # print(reg.intercept_)
# #
