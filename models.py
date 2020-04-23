import tensorflow as tf
from keras_vggface.models import *
from resnet_models import resnet


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
        # self.fcw = tf.keras.layers.Dense(params['num_neighbors']-1, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)
        w_init = tf.random.uniform(shape=(params['batch_size'], params['num_neighbors']-1), maxval=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess(inputs)
        features = self.feature_extractor(x)
        out = {"z": features}

        if self.params['use_weights']:
            # w = self.fcw(features)
            out["w"] = tf.nn.softmax(self.w, axis=1)
        else:
            out["w"] = tf.ones((out["z"].shape[0], self.params['num_neighbors']-1))
        return out


class resnet18(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super(resnet18, self).__init__(**kwargs)
        self.params = params
        self.base_model = resnet.resnet_18(res=params['image_size'])
        self.feature_extractor = tf.keras.Model(self.base_model.input, self.base_model.layers[1].output)
        self.feat_dim = self.feature_extractor.output_shape[-1]
        # self.fcw = tf.keras.layers.Dense(params['num_neighbors']-1, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)
        w_init = tf.random.uniform(shape=(params['batch_size'], params['num_neighbors']-1), maxval=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        features = self.feature_extractor(inputs)
        out = {"z": features}

        if self.params['use_weights']:
            # w = self.fcw(features)
            out["w"] = tf.nn.softmax(self.w, axis=1)
        else:
            out["w"] = tf.ones((out["z"].shape[0], self.params['num_neighbors']-1))
        return out
