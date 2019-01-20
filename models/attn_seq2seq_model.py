from keras import regularizers
from keras.layers import Bidirectional, Dense
from keras.layers import LSTM, CuDNNLSTM, GRU
from keras.layers import TimeDistributed, Dropout, concatenate

from configs import available_RNN
from layers import EncoderDecoderAttnLayer
from models import BasicModel


# TODO: Feed concat of final state of forward and backward pass of encoder to each decoding timestep.
# TODO: Feed avg pooling of output seq of encoder to each decoding timestep.
class AttnSeq2SeqModel(BasicModel):
    def __init__(self):
        super(AttnSeq2SeqModel, self).__init__()

    def _do_build(self, x_in_vec_seq, x_in_id_seq, x_mask,
                  y_in_vec_seq, y_in_id_seq, y_mask):
        p_dropout = self.hyperparams.p_dropout

        # Get done => Freely switch RNN network.
        rnn_class_name = ''
        use_which_rnn = self.hyperparams.rnn
        if use_which_rnn == available_RNN[0]:
            UsedRNN = LSTM
        elif use_which_rnn == available_RNN[1]:
            UsedRNN = GRU
        elif use_which_rnn == available_RNN[2]:
            UsedRNN = CuDNNLSTM
        else:
            UsedRNN = LSTM

        # encoder
        x = Dropout(p_dropout, name='x_in_dropout')(x_in_vec_seq)
        for i in range(self.hyperparams.encoder_layer_num):
            this_rnn = UsedRNN(self.hyperparams.hidden_state_dim, return_sequences=True,
                               kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                               recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                               bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                               activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
            rnn_class_name = this_rnn.__class__.__name__
            x = Bidirectional(this_rnn,
                              name=str(i+1)+'th_encoder_retseq_bi_'+rnn_class_name)(x)
            x = Dropout(p_dropout, name=str(i+1) + 'th_encoder_dropout')(x)
        # Projection_layer => to make sure dim of encoder outputs match decoder hidden vector.
        # The activation function should not be used here
        x_hidden_seq = TimeDistributed(Dense(self.hyperparams.context_vec_dim,
                                             name="projection_layer",
                                             kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                                             bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                                             activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                                             )
                                       )(x)

        # decoder
        y = Dropout(p_dropout, name='y_in_dropout')(y_in_vec_seq)
        for i in range(self.hyperparams.decoder_layer_num):
            y = UsedRNN(self.hyperparams.hidden_state_dim, return_sequences=True,
                        name=str(i+1)+'th_decoder_retseq_uni_'+rnn_class_name,
                        kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                        recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                        bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                        activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                        )(y)
            y = Dropout(p_dropout, name=str(i+1) + 'th_decoder_dropout')(y)
        y_hidden_seq = y

        context_vec_seq = EncoderDecoderAttnLayer(name='encoder_decoder_attn_layer'
                                                  )([x_hidden_seq, y_hidden_seq,
                                                     x_mask, y_mask])
        middle = concatenate([y_hidden_seq, context_vec_seq], axis=-1)
        # Follow Luong's work, use `tanh` activation function here.
        attn_hidden_seq = Dense(self.hyperparams.hidden_state_dim, activation='tanh',
                                name='attn_hidden_dense_layer',
                                kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                                bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                                activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                                )(middle)

        return attn_hidden_seq
