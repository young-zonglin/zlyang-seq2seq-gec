import sys

from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop

model_name_abbr_full = {'AS2SModel': 'AttnSeq2SeqModel'}
model_name_full_abbr = {v: k for k, v in model_name_abbr_full.items()}
available_models = ['AS2SModel']


def get_hyperparams(model_name):
    if model_name == available_models[0]:
        return AttnSeq2SeqHParams()
    else:
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() func, model_name value error.')


class LRSchedulerDoNothing(Callback):
    def __init__(self):
        super(LRSchedulerDoNothing, self).__init__()


available_len_mode = ['fixed', 'full_len', 'batch_len']


class BasicHParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        # Get done => Three len mode available.
        # Use fixed len mode due to memory including graphics card limitations.
        self.len_mode = available_len_mode[0]
        self.input_len = 20
        self.output_len = self.input_len
        self.keep_token_num = 10000
        self.token_vec_dim = 300

        self.encoder_layer_num = 1
        self.decoder_layer_num = 1
        self.hidden_state_dim = self.token_vec_dim
        self.context_vec_dim = self.hidden_state_dim

        self.p_dropout = 0.5

        self.batch_size = 128  # Integer multiple of 32

        self.optimizer = Adam()
        self.lr_scheduler = LRSchedulerDoNothing()

        self.kernel_l2_lambda = 0
        self.recurrent_l2_lambda = 0
        self.bias_l2_lambda = 0
        self.activity_l2_lambda = 0

        self.early_stop_monitor = 'val_loss'
        self.early_stop_mode = 'auto'
        # 10 times waiting is not enough.
        # Maybe 20 is a good value.
        self.early_stop_patience = 20
        self.early_stop_min_delta = 1e-4

        self.train_epoch_times = 1000

        self.pad = 'pre'
        self.cut = 'pre'

    def __str__(self):
        ret_info = list()
        ret_info.append('optimizer: ' + str(self.optimizer) + '\n')
        ret_info.append('lr scheduler: ' + str(self.lr_scheduler) + '\n\n')

        ret_info.append('len mode: ' + str(self.len_mode) + '\n')
        if self.len_mode == available_len_mode[0]:
            ret_info.append('input length: ' + str(self.input_len) + '\n')
            ret_info.append('output length: ' + str(self.output_len) + '\n')
        ret_info.append('keep token num: ' + str(self.keep_token_num) + '\n')
        ret_info.append('token vec dim: ' + str(self.token_vec_dim) + '\n\n')

        ret_info.append('encoder depth: ' + str(self.encoder_layer_num) + '\n')
        ret_info.append('decoder depth: ' + str(self.decoder_layer_num) + '\n')
        ret_info.append('hidden state dim: ' + str(self.hidden_state_dim) + '\n')
        ret_info.append('context vec dim: ' + str(self.context_vec_dim) + '\n\n')

        ret_info.append('dropout proba: ' + str(self.p_dropout) + '\n\n')

        ret_info.append('batch size: '+str(self.batch_size)+'\n\n')

        ret_info.append('kernel l2 lambda: ' + str(self.kernel_l2_lambda) + '\n')
        ret_info.append('recurrent l2 lambda: ' + str(self.recurrent_l2_lambda) + '\n')
        ret_info.append('bias l2 lambda: ' + str(self.bias_l2_lambda) + '\n')
        ret_info.append('activity l2 lambda: ' + str(self.activity_l2_lambda) + '\n\n')

        ret_info.append('early stop monitor: ' + str(self.early_stop_monitor) + '\n')
        ret_info.append('early stop mode: ' + str(self.early_stop_mode) + '\n')
        ret_info.append('early stop patience: ' + str(self.early_stop_patience) + '\n')
        ret_info.append('early stop min delta: ' + str(self.early_stop_min_delta) + '\n\n')

        ret_info.append('train epoch times: ' + str(self.train_epoch_times) + '\n\n')

        ret_info.append("pad: " + self.pad + '\n')
        ret_info.append("cut: " + self.cut + '\n\n')
        return ''.join(ret_info)


available_RNN = ['LSTM', 'GRU', 'CuDNNLSTM']


class AttnSeq2SeqHParams(BasicHParams):
    """
    The best result is a val_accuracy of about xx.xx%.
    """
    def __init__(self):
        super(AttnSeq2SeqHParams, self).__init__()

        self.rnn = available_RNN[0]
        # Set `keep_token_num` properly, just filter out tokens thar appear only once.
        # Motivation:
        # 1. avoid label unbalanced problem;
        # 2. reduce the vocab size significantly;
        # 3. avoid too many unk symbols occur.
        # TODO: vocabulary of src and tgt side
        # TODO: automatic determine it value
        self.keep_token_num = 138000

        self.encoder_layer_num = 1
        self.decoder_layer_num = 1

        self.p_dropout = 0.0

        self.kernel_l2_lambda = 0
        self.recurrent_l2_lambda = 0
        self.bias_l2_lambda = 0
        self.activity_l2_lambda = 0

        # RMSprop + momentum => Adam
        self.optimizer = Adam()
        self.lr_scheduler = LRSchedulerDoNothing()

        self.early_stop_monitor = 'val_loss'
        # One iteration is particularly time consuming.
        # Maybe 5 times is more than enough.
        self.early_stop_patience = 5

        # The max value limited by memory including graphics card and vocab size.
        self.batch_size = 32

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append("use which RNN: " + self.rnn + '\n\n')
        super_str = super(AttnSeq2SeqHParams, self).__str__()
        return ''.join(ret_info) + super_str


if __name__ == '__main__':
    print(BasicHParams())
    print(AttnSeq2SeqHParams())
