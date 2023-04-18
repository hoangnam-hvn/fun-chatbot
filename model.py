from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.layers import *

from preprocess_data import maxlen, input_dim, X_train, y_train


input_shape = (maxlen,)

input_layer = Input(input_shape)

x = Embedding(input_dim=input_dim, output_dim=100, input_length=15, mask_zero=True)(input_layer)
x = LSTM(30, activation='relu', return_sequences=True)(x)
x = LSTM(60, activation='relu', recurrent_dropout=0.1)(x)
x = Dense(60, activation='relu')(x)

output_layer = Dense(50, activation='softmax')(x)

model = Model(input_layer, output_layer)

model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1)

model.save('funchatbot.h5')