from preprocessing import get_preprocessed_df
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import winsound
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np




# load the data with get_preprocessed_df
X_train, X_test, y_train, y_test, holdout = get_preprocessed_df()


# create a Gaussian Naive Bayes classifier
gnb = GaussianNB()


# use grid search to find the best features for the classifier
parameters = {'var_smoothing': np.logspace(0,-9, num=1000)}

clf = GridSearchCV(gnb, parameters, cv=5, verbose=1, n_jobs=-1, scoring='f1')

clf.fit(X_train, y_train)

# print the best parameters
print(clf.best_params_)

# print the best score
print(clf.best_score_)

