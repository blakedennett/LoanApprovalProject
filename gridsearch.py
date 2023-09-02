from preprocessing import get_preprocessed_df
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import winsound
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import winsound



def run_gridsearch(classifier, hyperparameters, cv=2):
    X_train, X_test, y_train, y_test, holdout = get_preprocessed_df()

    # Set up grid search
    clf = GridSearchCV(classifier, hyperparameters, cv=cv, verbose=1, n_jobs=-1, scoring='f1')

    # Fit data
    clf.fit(X_train, y_train)

    # Print the best parameters
    print("Best parameters:")
    print(clf.best_params_)

    # Print the best score
    print("Best F1 score:", clf.best_score_)







hyperparameters = {
            "max_depth": range(4, 32, 4),                           
            "min_samples_split": range(8, 15, 2),                  
            "min_samples_leaf": range(2, 5),                
            "criterion": ["gini", "entropy", "log_loss"], 
            "max_leaf_nodes": range(37, 90, 5),                     
            "min_impurity_decrease": [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],              
            "min_weight_fraction_leaf": [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.009, 0.01]           
        }

# create classifier 
decisiontree = DecisionTreeClassifier()

# set up grid search
clf = GridSearchCV(decisiontree, hyperparameters, cv=2, verbose=1, n_jobs=-1, scoring='f1')

# run_gridsearch(decisiontree, hyperparameters)


# Best parameters:
# {'criterion': 'entropy', 'max_depth': 4, 'max_leaf_nodes': 37, 'min_impurity_decrease': 0.006, 'min_samples_leaf': 2, 'min_samples_split': 8, 'min_weight_fraction_leaf': 0}
# Best F1 score: 0.7670602692617042







hyperparameters = {
            "max_depth": range(4, 15, 4),                 
            "min_split_loss": range(2, 9, 2),           
            "min_child_weight": range(3, 12, 3),          
            "subsample": [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],                 
            "reg_lambda": range(2, 7),                 
            "reg_alpha": range(1, 5),                  
            "learning_rate": [0.001, 0.002, 0.003, 0.005, 0.008, 0.099]             
        }

# create classifier 
xgb = XGBClassifier()

# run_gridsearch(xgb, hyperparameters)


# Best parameters:
# {'learning_rate': 0.001, 'max_depth': 4, 'min_child_weight': 3, 'min_split_loss': 6, 'reg_alpha': 2, 'reg_lambda': 3, 'subsample': 0.9}
# Best F1 score: 0.7669218710047911







gnb = GaussianNB()

# use grid search to find the best features for the classifier
parameters = {'var_smoothing': np.logspace(0,-9, num=1000)}

# run_gridsearch(gnb, parameters, 20)


# Best parameters:
# {'var_smoothing': 1.0}
# Best F1 score: 0.766036205723853








hyperparameters = {
            "max_depth": range(15, 31, 5),                    
            "min_samples_split": range(4, 9, 2),             
            "min_samples_leaf": range(1, 7, 2),             
            "bootstrap": [True, False],
            "warm_start": [True, False],
            "min_weight_fraction_leaf": (0.02, 0.026, 0.035, 0.05),    
            "n_estimators": [150, 250, 350, 450],
            'criterion': ['gini', 'entropy', 'log_loss']
        }

rfc = RandomForestClassifier()

# run_gridsearch(rfc, hyperparameters)


# Best parameters:
# {'bootstrap': False, 'criterion': 'log_loss', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 4, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 250, 'warm_start': False}
# Best F1 score: 0.7636556876292391











# set up hyperparameters for kneighbors classifier
hyperparameters = {
        "weights": ['uniform', 'distance'],
        "n_neighbors": range(4, 33),        
        "p": range(1, 3),                   
        "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

# create classifier 
kneighbors = KNeighborsClassifier()

# run_gridsearch(kneighbors, hyperparameters, 20)

# Best parameters:
# {'algorithm': 'auto', 'n_neighbors': 31, 'p': 2, 'weights': 'uniform'}
# Best F1 score: 0.7432682909080076