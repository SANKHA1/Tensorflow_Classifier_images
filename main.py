import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
inputs = keras.Input(shape=(784,))

dense = layers.Dense(64, activation="relu")
x = dense(inputs)

x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
filepath = 'model_checkpoint/file.hdf5'
model = keras.Model(inputs=inputs, outputs=outputs, name="flowers")



# X, y= keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
callback =  ModelCheckpoint('model.{epoch:02d}-{val_loss:0.2f}.h5', verbose = 1,
                                   monitor = "val_accuracy",mode='max',save_best_only=True)
model.load_weights('model.06-0.14.h5')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data =(x_test, y_test) , callbacks=[callback])

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

