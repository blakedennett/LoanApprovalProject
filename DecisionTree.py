from preprocessing import get_preprocessed_df
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import random
import winsound
import time
from xgboost import XGBClassifier
import multiprocessing as mp
from sklearn.naive_bayes import GaussianNB



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



def decision_tree(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)

        max_depth = random.randint(4, 32)                           # 4 to 32 levels
        min_samples_split = random.randint(2, 15)                   # 2 to 15 samples
        min_samples_leaf = random.randint(1, 15)                    # 1 to 15 samples
        criterion = random.choice(["gini", "entropy", "log_loss"])  # gini, entropy, log_loss
        class_weight = random.choice(["balanced", None])            # balanced, None
        max_leaf_nodes = random.randint(20, 80)                     # 20 to 80 nodes
        min_impurity_decrease = random.uniform(0, 0.1)              # 0 to 0.1
        min_weight_fraction_leaf = random.uniform(0, 0.1)           # 0 to 0.1


        hyperparameters = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": None,
            "criterion": criterion,
            "random_state": 42,
            "class_weight": class_weight,
            "splitter": "best",
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
        }

        model = DecisionTreeClassifier(**hyperparameters)
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1
            holdout_true = holdout[' loan_status']
            holdout.drop(columns=[' loan_status'], inplace=True)
            holdout_pred = model.predict(holdout[chosen_features])
            val_acc = accuracy_score(y_test, y_pred)
            holdout_f1 = f1_score(holdout_true, holdout_pred)
            holdout_acc = accuracy_score(holdout_true, holdout_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, "DecisionTree")

            winsound.Beep(350, 600)





def xgb(best_so_far=0.64):

    while True:

        x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

        n_features, chosen_features = get_features(holdout)


        max_depth = random.randint(4, 32)                           # 4 to 32 levels
        learning_rate = random.uniform(0, 0.1)                      # 0 to 0.1
        min_split_loss = random.randint(0, 20)                      # 0 to 20
        min_child_weight = random.randint(0, 20)                    # 0 to 20
        subsample = random.uniform(0.1, 1)                          # 0.1 to 1
        reg_lambda = random.randint(1, 5)                           # 1 to 5
        reg_alpha = random.randint(1, 5)                            # 1 to 5
        

        hyperparameters = {
            "max_depth": max_depth,
            "random_state": 42,
            "min_split_loss": min_split_loss,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "learning_rate": learning_rate
        }

        model = XGBClassifier(**hyperparameters)
        model.fit(x_train[chosen_features], y_train)
        y_pred = model.predict(x_test[chosen_features])
        f1 = f1_score(y_test, y_pred)
        if f1 > best_so_far:
            best_so_far = f1
            holdout_true = holdout[' loan_status']
            holdout.drop(columns=[' loan_status'], inplace=True)
            holdout_pred = model.predict(holdout[chosen_features])
            val_acc = accuracy_score(y_test, y_pred)
            holdout_f1 = f1_score(holdout_true, holdout_pred)
            holdout_acc = accuracy_score(holdout_true, holdout_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, hyperparameters, n_features, chosen_features, "XGB")

            winsound.Beep(800, 600)






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
            holdout_true = holdout[' loan_status']
            holdout.drop(columns=[' loan_status'], inplace=True)
            holdout_pred = model.predict(holdout[chosen_features])
            val_acc = accuracy_score(y_test, y_pred)
            holdout_f1 = f1_score(holdout_true, holdout_pred)
            holdout_acc = accuracy_score(holdout_true, holdout_pred)

            record_results(f1, val_acc, holdout_f1, holdout_acc, None, n_features, chosen_features, "GaussianNB")

            winsound.Beep(350, 1800)






if __name__ == '__main__':

    p1 = mp.Process(target=decision_tree, args=(0.771778,))
    p2 = mp.Process(target=xgb, args=(0.7748008689,))
    p3 = mp.Process(target=gaussian_nb, args=(0.7722342733188721,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()