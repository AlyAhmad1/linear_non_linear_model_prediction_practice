"""
    https://github.com/codebasics/py/blob/master/ML/13_kmeans/Exercise/13_kmeans_exercise.ipynb

"""


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris_dataset = load_iris()

# x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.2)

df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

# plt.scatter(df['petal width (cm)'], df['petal length (cm)'], color='red')

# m1 = MinMaxScaler()
# df['petal_length_new'] = m1.fit_transform(df[['petal length (cm)']])
#
# m2 = MinMaxScaler()
# df['petal_width_new'] = m1.fit_transform(df[['petal width (cm)']])

# df.drop(['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'], axis=1, inplace=True)
df.drop(['sepal length (cm)','sepal width (cm)'], axis=1, inplace=True)
k_means_obj = KMeans(n_clusters=3)
df['cluster'] = k_means_obj.fit_predict(df)

df_1 = df[df['cluster'] == 0]
df_2 = df[df['cluster'] == 1]
df_3 = df[df['cluster'] == 2]


plt.scatter(df_1['petal length (cm)'], df_1['petal width (cm)'],color='blue')
plt.scatter(df_2['petal length (cm)'], df_2['petal width (cm)'],color='green')
plt.scatter(df_3['petal length (cm)'], df_3['petal width (cm)'],color='yellow')

plt.show()

