from preprocessing import get_preprocessed_df
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import math
from sklearn.metrics import f1_score, accuracy_score
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt
import numpy as np
from keras.callbacks import Callback



import time
start_time = time.time()

x_train, x_test, y_train, y_test, holdout = get_preprocessed_df(with_cibil=True)

num_features = x_train.shape[1]

model = Sequential()

model.add(Dense(units=16, input_dim=num_features, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=[Precision(), Recall()])

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

y_pred = model.predict(x_test)

y_pred = np.where(y_pred > 0.5, 1, 0)

print("Accuracy score:", accuracy_score(y_test, y_pred))

print("F1 score:", f1_score(y_test, y_pred))


