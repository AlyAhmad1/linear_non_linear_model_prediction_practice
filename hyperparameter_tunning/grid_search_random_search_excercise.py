"""
    learning link: https://github.com/codebasics/py/blob/master/ML/15_gridsearch/15_grid_search.ipynb
    exercise link: https://github.com/codebasics/py/blob/master/ML/15_gridsearch/Exercise/15_grid_search_cv_exercise.ipynb
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

data_set = datasets.load_digits()
print(dir(data_set))


x_train, x_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2)

# grid search tuning.
clf = GridSearchCV(RandomForestClassifier(), {'n_estimators':[20, 10,5]}, cv=5, return_train_score=False)
clf.fit(data_set.data, data_set.target)
print(clf.best_params_)
print(clf.best_score_)

# random-search tuning.
clf_1 = RandomizedSearchCV(RandomForestClassifier(), {'n_estimators':[20, 10,5]}, cv=5, return_train_score=False, n_iter=10)
clf_1.fit(x_train, y_train)
clf_1.fit(data_set.data, data_set.target)

print(clf_1.best_params_)
print(clf_1.best_score_)



