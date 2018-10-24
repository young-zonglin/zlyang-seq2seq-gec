from keras import backend as K
from keras.engine import Layer
from keras.layers import Activation, Add


class EncoderDecoderAttnLayer(Layer):
    def __init__(self, **kwargs):
        super(EncoderDecoderAttnLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x_hidden_seq, y_hidden_seq, x_mask, y_mask = inputs
        q, k, v = y_hidden_seq, x_hidden_seq, x_hidden_seq
        # mask operation => done
        y_mask = K.expand_dims(y_mask, -1)
        x_mask = K.expand_dims(x_mask, 1)
        mask = K.batch_dot(y_mask, x_mask, axes=[2, 1])
        mmask = (-1e+10) * (1-mask)
        attn = K.batch_dot(q, k, axes=[2, 2])
        attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        context_vec_seq = K.batch_dot(attn, v)
        return context_vec_seq * y_mask

    def compute_output_shape(self, input_shape):
        return input_shape[1]
