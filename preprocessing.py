
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE



def get_preprocessed_df(with_cibil=False, standard_scaling=False, filtered_features=False, reject_oversample=False):

    df = pd.read_csv(r'C:\Users\Blake Dennett\Downloads\Summer2023\loan_approval_dataset.csv')

    numerical_df = df.drop(columns=[' loan_status', ' education', ' self_employed'])

    std_factor = 3

    stdv_df = numerical_df.std()   # creates a series of standard deviations for each column
    avg_df = numerical_df.mean()

    upper_limits = avg_df + std_factor * stdv_df     # creates a series of upper limits for each column
    lower_limits = avg_df - std_factor * stdv_df

    numerical_cols = numerical_df.columns

    # Create a condition for numeric columns only
    condition = (numerical_df[numerical_cols] > upper_limits) | (numerical_df[numerical_cols] < lower_limits)

    # Update the values in the original DataFrame 'df' with the capped values
    df[numerical_cols] = df[numerical_cols].where(~condition, other=upper_limits, axis=0)


    for col in numerical_cols:
        df[col] = winsorize(df[col], limits=(0.03, 0.97))

    # get rid of negative values
    df[numerical_cols] = df[numerical_cols].applymap(lambda x: x if x >= 0 else 0)

    # handle ordinal, categorical features
    enc = OrdinalEncoder()
    enc.fit(df[['loan_id']])
    df[['loan_id']] = enc.transform(df[['loan_id']])


    collateral_df = df[[' residential_assets_value',  ' commercial_assets_value', ' bank_asset_value', ' luxury_assets_value']]
    df[' total_collateral'] = collateral_df.apply(lambda x: x.sum(), axis=1)

    df[' loan_coll_ratio'] = df[' loan_amount'] / df[' total_collateral']

    df[' loan_income_ratio'] = df[' loan_amount'] / df[' income_annum']

    df[' term_times_income'] = df[' loan_term'] * df[' income_annum']

    df[' col_times_term'] = df[' total_collateral'] * df[' loan_term']

    df[' lux_times_res'] = df[' luxury_assets_value'] * df[' residential_assets_value']

    df[' loan_status'] = np.where(df[' loan_status'] == " Approved", 1, 0)

    df.drop(columns=['loan_id'], inplace=True)
    if not with_cibil:
        df.drop(columns=[' cibil_score'], inplace=True)

    if filtered_features:
        df = df[[' loan_status', ' loan_income_ratio', ' loan_coll_ratio', ' cibil_score']]

    if not filtered_features:
        df = pd.get_dummies(df)

    holdout = df.sample(frac=0.1, random_state=42)

    x = df.drop(columns=[' loan_status'])
    y = df[' loan_status']


    # Initialize the splitter
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Perform the split
    for train_index, test_index in stratified_splitter.split(df, df[' loan_status']):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

    x_train = train_data.drop(columns=[' loan_status'])
    y_train = train_data[' loan_status']
    x_test = test_data.drop(columns=[' loan_status'])
    y_test = test_data[' loan_status']

    
    if reject_oversample:
        # Identify the 'rejected' samples in the training data
        rejected_indices = train_data[train_data[' loan_status'] == 0].index

        # Randomly sample some of the 'rejected' samples to oversample
        oversampled_indices = np.random.choice(rejected_indices, size=len(rejected_indices), replace=True)

        # Create a new DataFrame with the oversampled 'rejected' samples
        oversampled_data = df.iloc[oversampled_indices]

        # Concatenate the oversampled 'rejected' samples with the original training data
        train_data = pd.concat([train_data, oversampled_data])

        # Shuffle the training data to ensure randomness
        train_data = train_data.sample(frac=1, random_state=42)

        # Update x_train and y_train
        x_train = train_data.drop(columns=[' loan_status'])
        y_train = train_data[' loan_status']




    if standard_scaling:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
        x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])
        holdout[numerical_cols] = scaler.transform(holdout[numerical_cols])

    # x_train.drop(columns=[0], inplace=True)

    return x_train, x_test, y_train, y_test, holdout

x_train, x_test, y_train, y_test, holdout = get_preprocessed_df(with_cibil=True, reject_oversample=True)

# print(x_train.head())


x_train, x_test, y_train, y_test, holdout = get_preprocessed_df(with_cibil=True, reject_oversample=False)


# print(x_train.head())



# print(x_train.dtypes)
# print(y_train.dtypes)
# model = DecisionTreeClassifier(random_state=42)
# model.fit(x_train, y_train)

# feat_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=["Importance"])
# feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

# # Create the bar chart using Matplotlib
# plt.figure(figsize=(8, 6))
# plt.bar(feat_importances.index, feat_importances['Importance'], color='#21EB2B')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importances')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# def main():
#     plt.show()

# if __name__ == '__main__':
#     main()


def duplicate_rejects(x_train, y_train):

        def combine(dat1, dat2):
            return pd.concat([dat1, dat2])

        # isolate the rejects data
        rejects_df = y_train[y_train == 0]

        # combine the original with the duplicates
        y_train = combine(y_train, rejects_df)

        rejects_dict = {}

        # get indices from x_train rejects values
        rej_indices = list(rejects_df.index.values.tolist())

        # iterate through x_train, if it is a rejects index, add it to rejects dictionary
        for row in x_train.iterrows():
            if row[0] in rej_indices:
                # add row to dictionary
                rejects_dict[row[0]] = row[1]

        # turn the dictionary into a series
        x_rej_ser = pd.Series(rejects_dict)

        # combine the original with the duplicates
        x_train = combine(x_train, x_rej_ser)

        x_train.drop(columns=[0], inplace=True)

        return x_train, y_train

    # if reject_oversample:
    #     x_train, y_train = duplicate_rejects(x_train, y_train)