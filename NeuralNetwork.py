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
import numpy as np
from keras.callbacks import Callback



import time
start_time = time.time()

x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

x_train.drop(columns=[' no_of_dependents_0', ' no_of_dependents_1', ' no_of_dependents_2', ' no_of_dependents_3', ' no_of_dependents_4', ' no_of_dependents_5'], inplace=True)
x_test.drop(columns=[' no_of_dependents_0', ' no_of_dependents_1', ' no_of_dependents_2', ' no_of_dependents_3', ' no_of_dependents_4', ' no_of_dependents_5'], inplace=True)
holdout.drop(columns=[' no_of_dependents_0', ' no_of_dependents_1', ' no_of_dependents_2', ' no_of_dependents_3', ' no_of_dependents_4', ' no_of_dependents_5'], inplace=True)

num_features = x_train.shape[1]

def build_model(hp):
    model = Sequential()

    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])

    for i in range(hp.Int('num_layers', 1, 3)):
        hp_units = hp.Int(f'layer{i+1}', min_value=30, max_value=240, step=30)
        model.add(Dense(units=hp_units, input_dim=num_features, activation=hp_activation))
        # use hyperband to tune dropout rate
        hp_dropout_rate = hp.Choice(f'dropout{i+1}', values=[0.0, 0.01, 0.001, 0.0001])
        model.add(Dropout(rate=hp_dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001, 0.0001])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=hp_learning_rate), metrics=[Precision(), Recall()])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=20,
                     factor=3,
                     project_name='Hyperband_log4'
                     )


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def lr_schedule(epoch, lr):
    initial_learning_rate = .01
    decay_rate = 0.99
    epoch_rate = 2
    return initial_learning_rate * math.pow(decay_rate, math.floor(epoch/epoch_rate))

lr_callback = LearningRateScheduler(lr_schedule, verbose=1)

class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = np.round(self.model.predict(x_val))
        f1 = f1_score(y_val, y_pred)
        logs['val_f1_score'] = f1

f1_callback = F1ScoreCallback(validation_data=(x_test, y_test))

tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[stop_early, lr_callback, f1_callback])

print(tuner.results_summary())

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print()
print()

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')} 
# """)


hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=[lr_callback, f1_callback])

val_f1_per_epoch = history.history['val_f1_score']
best_epoch = val_f1_per_epoch.index(max(val_f1_per_epoch)) + 1
print('-----------------------------------------------Best epoch: %d----------------------------' % (best_epoch,))


model = tuner.hypermodel.build(best_hps)

# Retrain the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=best_epoch, callbacks=[lr_callback, f1_callback])



# After training, compute F1 score on holdout set
holdout_probabilities = model.predict(holdout.drop(columns=[' loan_status']))
holdout_pred = (holdout_probabilities > 0.5).astype(int)  # Apply threshold to classify
holdout_true = holdout[' loan_status']
holdout_f1 = f1_score(holdout_true, holdout_pred)

probabilities = model.predict(x_test)
pred = (probabilities > 0.5).astype(int)  # Apply threshold to classify
true = y_test
f1 = f1_score(true, pred)

print("F1 score on Validation:", f1)
print("F1 score on Holdout:", holdout_f1)
print("--- %s seconds ---" % (time.time() - start_time))












































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