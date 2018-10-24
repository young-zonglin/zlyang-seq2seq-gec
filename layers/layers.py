from keras import backend as K
from keras.engine import Layer


class GetPadMask(Layer):
    def __init__(self, **kwargs):
        super(GetPadMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Mask placeholder to zero.
        :param inputs: word id seq.
        :param kwargs:
        :return: mask seq.
        """
        word_id_seq = inputs
        mask = K.cast(K.greater(word_id_seq, 0), 'float32')
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape
