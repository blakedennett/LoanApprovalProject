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

def build_model(hp):
    model = Sequential()

    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])

    for i in range(hp.Int('num_layers', 1, 4)):
        hp_units = hp.Int(f'layer{i+1}', min_value=15, max_value=315, step=30)
        model.add(Dense(units=hp_units, input_dim=num_features, activation=hp_activation))
        # use hyperband to tune dropout rate
        hp_dropout_rate = hp.Choice(f'dropout{i+1}', values=[0.0, 0.01, 0.001, 0.0001, 0.00001])
        model.add(Dropout(rate=hp_dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001, 0.0001, 0.00001])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=hp_learning_rate), metrics=[Precision(), Recall()])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=20,
                     factor=3,
                     project_name='Hyperband_log2.0'
                     )


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def lr_schedule(epoch, lr):
    initial_learning_rate = .01
    decay_rate = 0.96
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

print('---------------------------------------------Best Hyperparameters------------------------------------------------')
print(f'Number of Layers: {best_hps.get("num_layers")}')
print(f'Learning Rate: {best_hps.get("learning_rate")}')
print(f'Activation: {best_hps.get("activation")}')
for i in range(best_hps.get("num_layers")):
    print(f'Units in Layer {i+1}: {best_hps.get("layer"+str(i+1))}')
    print(f'Dropout Rate in Layer {i+1}: {best_hps.get("dropout"+str(i+1))}')

print()


hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=[lr_callback, f1_callback])

val_f1_per_epoch = history.history['val_f1_score']
best_epoch = val_f1_per_epoch.index(max(val_f1_per_epoch)) + 1
print('-----------------------------------------------Best epoch: %d----------------------------' % (best_epoch,))


model = tuner.hypermodel.build(best_hps)

# Retrain the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, callbacks=[lr_callback, f1_callback])



# After training, compute F1 score on holdout set
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
print("--- %s seconds ---" % (time.time() - start_time))















































# Plot loss
# plt.clf()
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()