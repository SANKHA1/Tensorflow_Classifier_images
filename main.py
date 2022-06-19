import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from model_class import Resnet50_tf
from keras.layers import Dense
from keras.models import Sequential
from keras.applications.resnet import ResNet50
from keras.optimizer_experimental.sgd import SGD
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout,Activation,GlobalMaxPool2D,GlobalAvgPool2D,Input

aug = ImageDataGenerator(preprocessing_function=preprocess_input)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)

x_train = np.repeat(x_train, 3, axis=-1)

x_test = np.expand_dims(x_test, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test , num_classes=10)
pretrained = ResNet50(include_top=False,input_shape=(32,32,3), weights='imagenet',
                                   classes=10)
pretrained.trainable=False
def create_model():
    model = Sequential()
    model.add(layers.Resizing(height=32, width=32, interpolation='nearest'))
    model.add(pretrained)
    model.add(Dense(10))
    return model



if __name__  ==  "__main__":




    # resnet_model = Resnet50_tf(10)
    resnet_model = create_model()
    BATCH_SIZE = 8
    N_EPOCHS = 2
    # resnet_model.pretrained.trainable = False
    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=tf.keras.metrics.CategoricalAccuracy())

    resnet_model.fit(aug.flow(x_train, y_train, BATCH_SIZE),validation_data=(x_test, y_test),
                batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                steps_per_epoch=x_train.shape[0]//BATCH_SIZE)
    resnet_model.summary()
    print("Model Training Completed ------ >>>>>>>")
    tf.saved_model.save(resnet_model, 'exported_model/Resnet50')
