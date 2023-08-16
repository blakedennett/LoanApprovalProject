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



import time
start_time = time.time()

x_train, x_test, y_train, y_test, holdout = get_preprocessed_df()

num_features = x_train.shape[1]

model = Sequential()
model.add(Dense(64, input_dim=num_features, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.8))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))  # Use sigmoid activation for binary classification

optimizer = Adam(learning_rate=0.1)
model.compile(loss='binary_crossentropy', metrics=[Precision(), Recall()])

def lr_schedule(epoch, lr):
    initial_learning_rate = 0.1
    decay_rate = 0.96
    epoch_rate = 5
    return initial_learning_rate * math.pow(decay_rate, math.floor(epoch/epoch_rate))

lr_callback = LearningRateScheduler(lr_schedule, verbose=1)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64, callbacks=[lr_callback])

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


# Plot loss
plt.clf()
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plt.show()