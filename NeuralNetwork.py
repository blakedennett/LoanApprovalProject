from preprocessing import get_preprocessed_df
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import math
from sklearn.metrics import f1_score
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt


import time
start_time = time.time()

x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

num_features = x_train.shape[1]

def build_model(hp):
    model = Sequential()

    hp_units = hp.Int('units', min_value=30, max_value=60, step=30)
    model.add(Dense(units=hp_units, input_dim=num_features, activation='relu'))

    hp_dropout_rate = hp.RealInterval('dropout_rate', min_value=0.001, max_value=0.1, step=0.001)
    model.add(Dropout(rate=hp_dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=hp_learning_rate), metrics=[Precision(), Recall()])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=5,
                     factor=3,
                     )


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def lr_schedule(epoch, lr):
    initial_learning_rate = 0.01
    decay_rate = decay_rate
    epoch_rate = epoch_rate
    return initial_learning_rate * math.pow(decay_rate, math.floor(epoch/epoch_rate))

lr_callback = LearningRateScheduler(lr_schedule, verbose=1)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[stop_early, lr_callback])


# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print()
print()

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')} and the best dropout rate is .
""")

# print(best_hps.get('dropout_rate'))


















































hyperparameters = {
    "batch_size": [64, 80],
    "epochs": [5, 10],
    "learning_rate": [0.1, 0.01],
    'dropout_rate': [0.2, 0.9],
    'num_layers': [1, 2],
    'num_neurons': [32, 64]
}



    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[lr_callback])


 #     # After training, compute F1 score on holdout set
    # holdout_probabilities = model.predict(holdout.drop(columns=[' loan_status']))
    # holdout_pred = (holdout_probabilities > 0.5).astype(int)  # Apply threshold to classify
    # holdout_true = holdout[' loan_status']
    # holdout_f1 = f1_score(holdout_true, holdout_pred)

    # probabilities = model.predict(x_test)
    # pred = (probabilities > 0.5).astype(int)  # Apply threshold to classify
    # true = y_test
    # f1 = f1_score(true, pred)

    # print("F1 score on Validation:", f1)
    # print("F1 score on Holdout:", holdout_f1)
    # print("--- %s seconds ---" % (time.time() - start_time))


# Plot loss
# plt.clf()
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()