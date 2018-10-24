from keras import regularizers
from keras.layers import Bidirectional, Dense
from keras.layers import TimeDistributed, Dropout, concatenate, LSTM

from layers import EncoderDecoderAttnLayer, GetPadMask
from models import BasicModel


class AttnSeq2SeqModel(BasicModel):
    def __init__(self):
        super(AttnSeq2SeqModel, self).__init__()

    def _do_build(self, x_in_vec_seq, x_in_id_seq, y_in_vec_seq, y_in_id_seq):
        p_dropout = self.hyperparams.p_dropout

        # encoder
        x = Dropout(p_dropout, name='x_in_dropout')(x_in_vec_seq)
        for i in range(self.hyperparams.encoder_layer_num):
            this_lstm = LSTM(self.hyperparams.hidden_state_dim, return_sequences=True,
                             kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                             recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                             bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                             activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
            x = Bidirectional(this_lstm, name=str(i+1)+'th_encoder_retseq_bilstm')(x)
            x = Dropout(p_dropout, name=str(i+1) + 'th_encoder_dropout')(x)
        # to make sure dim of encoder outputs match decoder hidden vector
        x_hidden_seq = TimeDistributed(Dense(self.hyperparams.context_vec_dim, activation='relu',
                                             name="encoder_dense_layer",
                                             kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                                             bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                                             activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
                                       )(x)

        # decoder
        y = Dropout(p_dropout, name='y_in_dropout')(y_in_vec_seq)
        for i in range(self.hyperparams.decoder_layer_num):
            y = LSTM(self.hyperparams.hidden_state_dim, return_sequences=True,
                     name=str(i+1)+'th_decoder_retseq_unilstm',
                     kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                     recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                     bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                     activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                     )(y)
            y = Dropout(p_dropout, name=str(i+1) + 'th_decoder_dropout')(y)
        y_hidden_seq = y

        get_pad_mask = GetPadMask()
        x_mask = get_pad_mask(x_in_id_seq)
        y_mask = get_pad_mask(y_in_id_seq)
        context_vec_seq = EncoderDecoderAttnLayer(name='encoder_decoder_attn_layer'
                                                  )([x_hidden_seq, y_hidden_seq, x_mask, y_mask])
        middle = concatenate([y_hidden_seq, context_vec_seq], axis=-1)
        attn_hidden_seq = Dense(self.hyperparams.hidden_state_dim, activation='relu',
                                name='attn_hidden_dense_layer')(middle)

        return attn_hidden_seq
