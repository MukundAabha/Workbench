# Sequential model

import tensorflow as tf
from tensorflow import keras

NB_classes = 10  # 10 artificial neurons
RESHAPED = 784  # 784 input variables
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_classes,
                             input_shape=(RESHAPED,), kernal_initializer='zeros',
                             name='dense_layer', activation='softmax'))
