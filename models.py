import tensorflow as tf
from keras_vggface.models import *


def preprocess_senet50(x, mean=(91.4953, 103.8827, 131.0912)):
    x = x[..., ::-1]
    x = x * 255.0
    return x - mean


class Classifier(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.params = params
        self.preprocess = tf.keras.layers.Lambda(preprocess_senet50)
        self.feature_extractor = SENET50(include_top=False, weights='vggface',
                                         input_shape=(params['image_size'], params['image_size'], 3),
                                         pooling='avg')
        self.feat_dim = self.feature_extractor.output_shape[-1]
        self.fcw = tf.keras.layers.Dense(params['num_neighbors']-1, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)

    def call(self, inputs, training=None, mask=None):
        features = self.feature_extractor(inputs)
        out = {"z": features}

        if self.params['use_weights']:
            w = self.fcw(features)
            out["w"] = w
        else:
            out["w"] = tf.ones_like(self.params['num_neighbors']-1)
        return out


class Net(tf.keras.Model):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        self.inputs = tf.keras.layers.Input((None, params['num_samples']))
        self.fc1 = tf.keras.layers.Dense(params['embedding_dim'], kernel_initializer='he_normal', use_bias=False)
        self.fc2 = tf.keras.layers.Dense(params['embedding_dim'], kernel_initializer='he_normal', use_bias=False)
        self.fcw = tf.keras.layers.Dense(1, kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs, training=None, mask=None):
        x = tf.keras.layers.ReLU()(self.fc1(inputs))
        z = tf.keras.layers.ReLU()(self.fc2(x))

        out = {"z": z}

        if self.params['use_weights']:
            w = self.fcw(x)
            out["w"] = w

        return out
