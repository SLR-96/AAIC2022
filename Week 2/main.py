import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.metrics import r2_score
from numpy.random import seed
import tensorflow as tf

seed(0)
tf.random.set_seed(0)

train_file = pd.read_csv('./MCI-RD-aaic-UIUF/MCIRD_aaic2021_train.csv', index_col='subscriber_ecid')
train_file.insert(train_file.shape[1] - 1, 'data_usage_volume', train_file.pop('data_usage_volume'))

week1_file = pd.read_csv('./MCI-RD-aaic-UIUF/MCIRD_aaic2021_test_week1_with_target.csv', index_col='subscriber_ecid')
week1_file.insert(week1_file.shape[1] - 1, 'data_usage_volume', week1_file.pop('data_usage_volume'))

week2_file = pd.read_csv('./MCI-RD-aaic-UIUF/MCIRD_aaic2021_test_week2.csv', index_col='subscriber_ecid')

people = np.unique(week2_file.index)

x = []
y = []
for person in people:
    person_train = train_file.loc[person]
    person_week1 = week1_file.loc[person]
    person_data = pd.concat([person_train, person_week1]).drop('day', axis=1).values
    data_dim = person_data.shape
    for i in range(data_dim[0]):
        for j in range(data_dim[1]):
            if np.isnan(person_data[i, j]):
                person_data[i, j] = np.nanmean(person_data[:, j])
    for i in range(1, 50):
        if i == 0:
            sequence = person_data.copy()
        else:
            sequence = person_data.copy()[:-i]
        pad = np.zeros([76 - sequence.shape[0], sequence.shape[1]])
        sequence = np.vstack((pad, sequence))
        output = sequence[-1, -1]
        sequence[-1, -1] = 0
        x.append(sequence)
        y.append(output)

x = np.asarray(x, dtype=np.float64)
y = np.asarray(y, dtype=np.float64)


# x, x_test, y, y_test = train_test_split(x, y, test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=0)

x_min = x_train.min(axis=(0, 1))
x_max = x_train.max(axis=(0, 1))
x_indices = np.where(x_min == x_max)[0]
for ind in x_indices:
    x_max[ind] += 1

np.save('./My Model/x_min.npy', x_min)
np.save('./My Model/x_max.npy', x_max)

x_train = (x_train - x_min) / (x_max - x_min)
x_val = (x_val - x_min) / (x_max - x_min)
# x_test = (x_test - x_min) / (x_max - x_min)


inp_layer = Input(shape=(None, x_train.shape[2]))
layer = LSTM(16, kernel_regularizer=l2(0.01), return_sequences=True)(inp_layer)
layer = LSTM(16, kernel_regularizer=l2(0.01), return_sequences=False)(layer)
out_layer = Dense(1, activation='linear')(layer)

my_model = Model(inp_layer, out_layer)
my_model.summary()

lr = CosineDecayRestarts(initial_learning_rate=0.001, first_decay_steps=30)
my_model.compile(optimizer=Adam(lr), loss=MSE)

cb = ModelCheckpoint('./Checkpoints/checkpoint', monitor='val_loss', save_best_only=True, save_weights_only=True)
results = my_model.fit(x_train, y_train, batch_size=512, epochs=3000, callbacks=[cb], validation_data=(x_val, y_val))
my_model.load_weights('./Checkpoints/checkpoint')
my_model.save('./My Model')

print('\nTest Results:')
my_model.evaluate(x_val, y_val)

y_train_pred = my_model.predict(x_train)
y_test_pred = my_model.predict(x_val)

plt.figure()
plt.semilogy(results.history['loss'])
plt.semilogy(results.history['val_loss'])
plt.title('Losses')
plt.legend(['Train Loss', 'Validation Loss'])
plt.xlabel('Epochs')
plt.ylabel('MSE')

plt.figure()
plt.plot(y_train)
plt.plot(y_train_pred)
plt.title('Train')
plt.legend(['Actual Data', 'Predicted Values'])
plt.xlabel('Day')
plt.ylabel('Usage Volume')

plt.figure()
plt.plot(y_val)
plt.plot(y_test_pred)
plt.title('Test')
plt.legend(['Actual Data', 'Predicted Values'])
plt.xlabel('Day')
plt.ylabel('Usage Volume')

r2_val = r2_score(y_true=y_val, y_pred=y_test_pred)
print('R2 score for validation:', r2_val)

r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
print('R2 score for train:', r2_train)
