"""
Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.

Now do some exploratory data analysis to figure out which variables have direct and clear impact on employee retention (i.e. whether they leave the company or continue to work)
Plot bar charts showing impact of employee salaries on retention
Plot bar charts showing correlation between department and employee retention
Now build logistic regression model using variables that were narrowed down in step 1
Measure the accuracy of the model

"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("employee_data.csv")

left_employee = df[df['left'] == 1]
retained_employee = df[df['left'] == 0]

# analyze data. ( this tells what impact of factors on employee retention )
temp_df = df
temp_df.drop('Department', axis=1, inplace=True)
temp_df.drop('salary', axis=1, inplace=True)
temp_df.groupby('left').mean()

# plot bar graph to check salary impact on employee retention
pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.show()


# plot bar graph to check department co-relation with employee retention
pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.show()

# these columns have direct impact on employee retention
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

# salary column is in string let's do one hot encodeing.
encoded_salaries = pd.get_dummies(subdf['salary'])

df_with_dummies = pd.concat([subdf,encoded_salaries],axis='columns')

# drop salary column bcz no more needed.
df_with_dummies.drop('salary',axis='columns',inplace=True)

X = df_with_dummies
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

# check model accuracy.
print(model.score(X_test,y_test))