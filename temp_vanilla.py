from preprocessing import get_preprocessed_df
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import math
from sklearn.metrics import f1_score, accuracy_score
from keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt
import numpy as np
from keras.callbacks import Callback
from keras.losses import CategoricalCrossentropy
from keras.losses import BinaryCrossentropy
import tensorflow_addons as tfa
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import shap
import time





x_train, x_test, y_train, y_test, holdout = get_preprocessed_df(with_cibil=True, standard_scaling=True)

num_features = x_train.shape[1]

model = Sequential()

model.add(Dense(units=315, input_dim=num_features, activation='relu'))
model.add(Dropout(rate=0.0))
model.add(Dense(units=45, input_dim=num_features, activation='relu'))
model.add(Dropout(rate=0.00001))
model.add(Dense(units=285, input_dim=num_features, activation='relu'))
model.add(Dropout(rate=0.01))
model.add(Dense(units=285, input_dim=num_features, activation='relu'))
model.add(Dropout(rate=0.0001))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss=BinaryCrossentropy(reduction='auto'), optimizer='rmsprop', metrics=[AUC()])

def lr_schedule(epoch, lr):
        initial_learning_rate = .1
        decay_rate = 0.90
        epoch_rate = 2
        return initial_learning_rate * math.pow(decay_rate, math.floor(epoch/epoch_rate))

lr_callback = LearningRateScheduler(lr_schedule, verbose=1)

for i in range(1):
    start_time = time.time()

    history = model.fit(x_train, y_train, epochs=6, batch_size=25, steps_per_epoch=25, validation_data=(x_test, y_test), verbose=1, callbacks=[lr_callback])


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
    print(f'num steps: {i}')
    print("--- %s seconds ---" % (time.time() - start_time))

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
# plt.show()

# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
# plt.show()


# sample_size = int(0.1 * len(x_test))
# x_test_subset = shap.sample(x_test, sample_size)

# # Create a function that predicts using your model (this is required by KernelExplainer)
# def predict_function(input_data):
#     return model.predict(input_data)

# # Create a SHAP explainer
# explainer = shap.KernelExplainer(predict_function, x_train)

# shap_values = explainer.shap_values(x_test_subset)

# shap.summary_plot(shap_values, x_test_subset, feature_names=x_train.columns)
