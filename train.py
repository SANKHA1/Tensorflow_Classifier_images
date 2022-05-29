import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from models.resnet50_tf import MyModel
from keras.optimizer_experimental.sgd import SGD
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(preprocessing_function=preprocess_input)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)

x_train = np.repeat(x_train, 3, axis=-1)

x_test = np.expand_dims(x_test, axis=-1)
x_test = np.repeat(x_test, 5, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test , num_classes=10)


if __name__  ==  "__main__":
    resnet_model = MyModel(10)
    BATCH_SIZE = 8
    N_EPOCHS = 5
    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=tf.keras.metrics.CategoricalAccuracy())

    resnet_model.fit(aug.flow(x_train, y_train, BATCH_SIZE),validation_data=(x_test, y_test),
                batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                steps_per_epoch=x_train.shape[0]//BATCH_SIZE)
    tf.saved_model.save(resnet_model, 'exported_model/Mymodel')


