import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
import tensorflow as tf
from keras.optimizers import Adam, Ftrl
from keras.metrics import AUC
import numpy as np
import matplotlib.pyplot as plt
import os


data_file_path = os.getenv("LOAN_DATA_PATH")
df = pd.read_csv(data_file_path)

collateral_df = df[[' residential_assets_value',  ' commercial_assets_value', ' bank_asset_value', ' luxury_assets_value']]

df[' loan_status'] = np.where(df[' loan_status'] == " Approved", 1, 0)

df[' total_collateral'] = collateral_df.apply(lambda x: x.sum(), axis=1)

df[' loan_coll_ratio'] = df[' loan_amount'] / df[' total_collateral']

df[' loan_income_ratio'] = df[' loan_amount'] / df[' income_annum']

df = df[[' loan_status', ' cibil_score', ' loan_income_ratio', ' loan_coll_ratio']]

holdout = df.sample(frac=0.1, random_state=42)
df.drop(holdout.index, inplace=True)

holdout_y = holdout[' loan_status']

holdout.drop(columns=[' loan_status'], inplace=True)


x = df.drop(columns=[' loan_status'])
# holdout.drop(columns=['loan_id'], inplace=True)
y = df[' loan_status']

x = pd.get_dummies(x)
holdout = pd.get_dummies(holdout)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



num_features = x_train.shape[1]

model = Sequential()

model.add(Dense(units=315, input_dim=num_features, activation='elu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss=BinaryCrossentropy(reduction='none'), optimizer=Ftrl(), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=16, steps_per_epoch=25, validation_data=(x_test, y_test), verbose=1)

holdout_probabilities = model.predict(holdout)
holdout_pred = (holdout_probabilities > 0.5).astype(int)  # Apply threshold to classify
holdout_true = holdout_y
holdout_f1 = f1_score(holdout_true, holdout_pred)
holdout_acc = accuracy_score(holdout_true, holdout_pred)
print(confusion_matrix(holdout_true, holdout_pred))

probabilities = model.predict(x_test)
pred = (probabilities > 0.5).astype(int)
true = y_test
f1 = f1_score(true, pred)
accuracy = accuracy_score(true, pred)
print(confusion_matrix(true, pred))

print("F1 score on Validation:", f1)
print("Accuracy on Validation:", accuracy)
print("F1 score on Holdout:", holdout_f1)
print("Accuracy on Holdout:", holdout_acc)


cm = confusion_matrix(holdout_true, holdout_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
disp.plot()

plt.show()