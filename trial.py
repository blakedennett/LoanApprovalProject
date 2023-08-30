import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, accuracy_score
from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from keras.optimizers import Adam
from keras.metrics import AUC





df = pd.read_csv(r'C:\Users\Blake Dennett\Downloads\Summer2023\loan_approval_dataset.csv')

holdout = df.sample(frac=0.1, random_state=42)


for col in df.columns:
    print(col)


x = df.drop(columns=[' loan_status', 'loan_id'])
y = df[' loan_status']

x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


num_features = x_train.shape[1]

model = Sequential()

model.add(Dense(units=315, input_dim=num_features, activation='tanh'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=[AUC()])

history = model.fit(x_train, y_train, epochs=10, batch_size=16, steps_per_epoch=25, validation_data=(x_test, y_test), verbose=1)

holdout_probabilities = model.predict(holdout.drop(columns=[' loan_status']))
holdout_pred = (holdout_probabilities > 0.5).astype(int)  # Apply threshold to classify
holdout_true = holdout[' loan_status']
holdout_f1 = f1_score(holdout_true, holdout_pred)
holdout_acc = accuracy_score(holdout_true, holdout_pred)

probabilities = model.predict(x_test)
pred = (probabilities > 0.5).astype(int)
true = y_test
f1 = f1_score(true, pred)
accuracy = accuracy_score(true, pred)

print("F1 score on Validation:", f1)
print("Accuracy on Validation:", accuracy)
print("F1 score on Holdout:", holdout_f1)
print("Accuracy on Holdout:", holdout_acc)


