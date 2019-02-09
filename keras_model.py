from keras.models import *
from keras.layers import *
from keras.metrics import top_k_categorical_accuracy

from get_data import Data

# 获取训练集和测试集
data = Data(100)
x_train, y_train, x_test, y_test = data.get_data_set()
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

model = Sequential()

model.add(Masking(mask_value=0., input_shape=(data.time_step, 1)))
model.add(LSTM(16, return_sequences=True, dropout=0.25))
model.add(LSTM(32, dropout=0.25))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(data.sku_all_num))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy', top_k_categorical_accuracy])

model.fit(x_train, y_train, batch_size=256, epochs=data.time_step, validation_data=(x_test, y_test))
