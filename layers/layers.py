from keras import backend as K
from keras.engine import Layer


class GetPadMask(Layer):
    def __init__(self, batch_size=None, **kwargs):
        super(GetPadMask, self).__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        """
        Mask placeholder to zero.
        :param inputs: token id seq or one-hot vec seq.
        :param kwargs:
        :return: mask seq.
        """
        input_ndim = K.ndim(inputs)
        if input_ndim == 2:
            token_id_seq = inputs
            mask = K.cast(K.greater(token_id_seq, 0), 'float32')
        elif input_ndim == 3:
            onehot_vec_seq = inputs
            middle_sum = K.reshape(K.sum(onehot_vec_seq, -1), (self.batch_size, -1))
            mask = K.cast(K.greater(middle_sum, 0), 'float32')
        else:
            raise ValueError('In class ' + self.__class__.__name__ +
                             ', input should be a 2D or 3D tensor.')
        return mask

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return input_shape
        elif len(input_shape) == 3:
            return input_shape[0], input_shape[1]
        else:
            raise ValueError('In class ' + self.__class__.__name__ +
                             ', input should be a 2D or 3D tensor.')

    def get_config(self):
        config = {'batch_size' : self.batch_size}
        base_config = super(GetPadMask, self).get_config()
        return dict(base_config.items() | config.items())
