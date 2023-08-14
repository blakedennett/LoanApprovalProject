from preprocessing import get_preprocessed_df
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import random
import winsound
import time
from xgboost import XGBClassifier
import multiprocessing as mp
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def vanilla_tree():
    x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    holdout_true = holdout[' loan_status']
    holdout.drop(columns=[' loan_status'], inplace=True)
    holdout_pred = model.predict(holdout)
    print("F1:", f1_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Holdout F1:", f1_score(holdout_true, holdout_pred))
    print("Holdout Accuracy:", accuracy_score(holdout_true, holdout_pred))


# vanilla_tree()


sure_features = [' loan_coll_ratio', ' loan_income_ratio', ' commercial_assets_value', ' bank_asset_value']



def get_features(holdout):
    unsure_df = holdout.drop(columns=sure_features)
    if ' loan_status' in unsure_df.columns:
        unsure_df.drop(columns=[' loan_status'], inplace=True)
    unsure_features = list(unsure_df.columns)

    n_features = random.randint(6, 18)       # 6 to 18 features
    chosen_features = random.shuffle(unsure_features)
    chosen_features = list(unsure_features[0:n_features - 4] + sure_features)

    return n_features, chosen_features



def record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, filename):

    path = r"C:\Users\Blake Dennett\Downloads\Summer2023\ml_results\\" + filename + ".txt" 

    with open(path, "a") as f:
        f.write('\n')
        f.write(f"Best F1 so far: {f1}\n")
        f.write(f"Accuracy = {val_acc}\n")
        f.write(f"Holdout F1 Score: {holdout_f1}\n")
        f.write(f"Holdout Accuracy = {holdout_acc}\n")
        if filename != "GaussianNB":
            f.write(f"Hyperparameters:\n")
            for key, value in hyperparameters.items():
                f.write(f"    {key}: {value}\n")
        f.write(f"Number of features: {n_features}\n")
        f.write(f"Chosen Features:\n")
        for feature in chosen_features:
            f.write(f"    {feature}\n")
        f.write("\n")


def get_results(holdout, model, chosen_features, y_test, y_pred):
    holdout_true = holdout[' loan_status']
    holdout.drop(columns=[' loan_status'], inplace=True)
    holdout_pred = model.predict(holdout[chosen_features])
    val_acc = accuracy_score(y_test, y_pred)
    holdout_f1 = f1_score(holdout_true, holdout_pred)
    holdout_acc = accuracy_score(holdout_true, holdout_pred)
    return val_acc, holdout_f1, holdout_acc





def decision_tree(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)

        hyperparameters = {
            "max_depth": random.randint(4, 30),                           # 4 to 32 levels
            "min_samples_split": random.randint(8, 15),                   # 2 to 15 samples
            "min_samples_leaf": random.randint(2, 7),                    # 1 to 15 samples
            "max_features": None,
            "criterion": random.choice(["gini", "entropy", "log_loss"]),
            "random_state": 42,
            "class_weight": random.choice(["balanced", None]),            # balanced, None
            "splitter": "best",
            "max_leaf_nodes": random.randint(35, 85),                     # 20 to 80 nodes
            "min_impurity_decrease": random.uniform(0, 0.05),              # 0 to 0.1
            "min_weight_fraction_leaf": random.uniform(0, 0.05)           # 0 to 0.1
        }

        model = DecisionTreeClassifier(**hyperparameters)
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1

            val_acc, holdout_f1, holdout_acc = get_results(holdout, model, chosen_features, y_test, y_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, "DecisionTree")

            winsound.Beep(350, 600)
            print('Decision Tree')






def xgb(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)        

        hyperparameters = {
            "max_depth": random.randint(4, 18),                 # 4 to 32 levels
            "random_state": 42,
            "min_split_loss": random.randint(0, 12),            # 0 to 12
            "min_child_weight": random.randint(1, 12),          # 1 to 12
            "subsample": random.uniform(0.1, 1),                # 0.1 to 1
            "reg_lambda": random.randint(2, 7),                 # 2 to 7
            "reg_alpha": random.randint(1, 3),                  # 1 to 3
            "learning_rate": random.uniform(0, 0.1)             # 0 to 0.1
        }

        model = XGBClassifier(**hyperparameters)
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1
            
            val_acc, holdout_f1, holdout_acc = get_results(holdout, model, chosen_features, y_test, y_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, "XGB")

            winsound.Beep(800, 600)
            print('XGB')






def gaussian_nb(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)


        model = GaussianNB()
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1
            
            val_acc, holdout_f1, holdout_acc = get_results(holdout, model, chosen_features, y_test, y_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, None, n_features, chosen_features, "GaussianNB")

            winsound.Beep(350, 1800)
            print('GaussianNB')






def random_forest(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)      

        hyperparameters = {
            "max_depth": random.randint(8, 32),                     # 4 to 32 levels
            "random_state": 42,
            "min_samples_split": random.randint(3, 8),              # 2 to 8
            "min_samples_leaf": random.randint(3, 7),               # 0 to 5
            "bootstrap": random.choice([True, True, True, True, False]),
            "warm_start": random.choice([True, False, False, False, False]),
            "min_weight_fraction_leaf": random.uniform(0, 0.1),     # 0 to 0.5
            "n_estimators": random.randint(75, 500),                # 10 to 500
            'criterion': random.choice(['gini', 'entropy', 'log_loss'])
        }

        model = RandomForestClassifier(**hyperparameters)
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1

            val_acc, holdout_f1, holdout_acc = get_results(holdout, model, chosen_features, y_test, y_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, "RandomForest")

            winsound.Beep(800, 1800)
            print('RandomForest')







def k_neighbors(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)          

        hyperparameters = {
            "weights": random.choice(['uniform', 'uniform', 'uniform', 'uniform', 'distance']),
            "n_neighbors": random.randint(16, 32),        # 4 to 30
            "p": random.randint(1, 2),                    # 1 to 2
            "algorithm": random.choice(['auto', 'auto', 'auto', 'auto', 'ball_tree', 'kd_tree', 'brute'])
        }

        model = KNeighborsClassifier(**hyperparameters)
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1
            
            val_acc, holdout_f1, holdout_acc = get_results(holdout, model, chosen_features, y_test, y_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, "KNeighbors")

            winsound.Beep(350, 200)
            winsound.Beep(350, 200)
            winsound.Beep(350, 200)
            print('KNeighbors')







# if __name__ == '__main__':

#     p1 = mp.Process(target=decision_tree, args=(0.7717782577393809,))
#     p2 = mp.Process(target=xgb, args=(0.7748008689,))
#     p3 = mp.Process(target=gaussian_nb, args=(0.7722342733188721,))
#     p4 = mp.Process(target=random_forest, args=(0.7759245830311821,))
#     p5 = mp.Process(target=k_neighbors, args=(0.7636092468307233,))

#     p1.start()
#     p2.start()
#     p3.start()
#     p4.start()
#     p5.start()

#     p1.join()
#     p2.join()
#     p3.join()
#     p4.join()
#     p5.join()


if __name__ == '__main__':

    p1 = mp.Process(target=decision_tree, args=(0.7717782577393809,))
    p2 = mp.Process(target=xgb, args=(0.772,))
    p3 = mp.Process(target=gaussian_nb, args=(0.7722342733188721,))
    p4 = mp.Process(target=random_forest, args=(0.7759245830311821,))
    p5 = mp.Process(target=k_neighbors, args=(0.7636092468307233,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()