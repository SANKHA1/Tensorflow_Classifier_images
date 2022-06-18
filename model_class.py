
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Dense, ReLU,Softmax

import keras.losses
import tensorflow
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout,Activation,GlobalMaxPool2D,GlobalAvgPool2D,Input
from keras.applications.resnet import ResNet50

class Resnet50_tf(Model):
    def __init__(self,num_classes,*args,  **kwargs):
        super(Resnet50_tf, self).__init__(**kwargs)
        self.pretrained = ResNet50(include_top=False,input_shape=(32,32,3), weights='imagenet',
                                   classes=num_classes)
        self.map = GlobalAvgPool2D()
        self.dense_1 = Dense(512)
        self.act_1 = Activation("relu")
        self.dropout = Dropout(0.2)
        self.dense_2 = Dense(num_classes)
        self.act_2 = Activation("softmax")

    def call(self, input_tensor):
        # inp = Input(shape=(None, None, 3))
        x = keras.layers.Lambda(lambda x: tf.image.resize(x, (32, 32)))(input_tensor)
        x = self.pretrained(x)
        x = self.map(x)
        x = self.dense_1(x)
        x = self.act_1(x)
        x = self.dense_2(x)
        x = self.act_2(x)
        return x
