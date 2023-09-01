import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np





df = pd.read_csv(r'C:\Users\Blake Dennett\Downloads\Summer2023\loan_approval_dataset.csv')

df[' loan_status'] = np.where(df[' loan_status'] == " Approved", 1, 0)

x = df.drop(columns=[' loan_status'])
y = df[' loan_status']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(len(x_train))
print(len(y_train))


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

    return x_train, y_train

x_train, y_train = duplicate_rejects(x_train, y_train)

print(len(x_train))
print(len(y_train))