import keras.losses
import tensorflow
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout,Activation,GlobalMaxPool2D,GlobalAvgPool2D
from keras.applications.resnet import ResNet50

class MyModel(Model):
    def __init__(self,num_classes,*args,  **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.pretrained = ResNet50(include_top=False, weights='imagenet',
                                   classes=num_classes)
        self.map = GlobalAvgPool2D()
        self.dense_1 = Dense(512)
        self.act_1 = Activation("relu")
        self.dropout = Dropout(0.2)
        self.dense_2 = Dense(num_classes)
        self.act_2 = Activation("softmax")

    def call(self, input_tensor):
        x = self.pretrained(input_tensor)
        x = self.map(x)
        x = self.dense_1(x)
        x = self.act_1(x)
        x = self.dense_2(x)
        x = self.act_2(x)
        return x








