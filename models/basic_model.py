import os
import time

import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, TimeDistributed, Dense
from keras.models import Model

from configs import EmbeddingParams
from configs import available_len_mode
from configs import base_params
from configs import corpus_name_full_abbr, model_name_full_abbr, embedding_name_full_abbr
from layers import GetPadMask
from models import beam_search
from utils import tools, reader

# Specify which GPU card to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TensorFlow显存管理，按需分配显存
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# 测试的时候batch size和seq len随意，token vec dim训练和应用时应一致
class BasicModel:
    def __init__(self, is_training=True):
        self.is_training = is_training

        self.hyperparams = None
        self.corpus_params = None
        self.embedding_params = None

        self.input_len = None
        self.output_len = None
        self.keep_token_num = None
        self.vocab_size = None
        self.token_vec_dim = None
        self.batch_size = None
        self.this_model_save_dir = None

        self.pretrained_embeddings_fname = None

        self.processed_url = None
        self.train_fname = None
        self.val_fname = None
        self.test_fname = None

        self.model = None
        self.embedding_matrix = None
        self.tokenizer = None
        self.id2token = None

        self.corpus_open_encoding = None
        self.embedding_open_encoding = None
        self.general_save_encoding = None

        self.pad = None
        self.cut = None

        self.total_samples_count = 0
        self.train_samples_count = 0
        self.val_samples_count = 0
        self.test_samples_count = 0

    # embedding_params can be None
    def setup(self, hyperparams, corpus_params, embedding_params):
        if embedding_params is None:
            embedding_params = EmbeddingParams()
        self.pretrained_embeddings_fname = embedding_params.pretrained_embeddings_url

        if embedding_params.char_level:
            self.processed_url = corpus_params.processed_url_char
            self.train_fname = corpus_params.train_url_char
            self.val_fname = corpus_params.val_url_char
            self.test_fname = corpus_params.test_url_char
        else:
            self.processed_url = corpus_params.processed_url_word
            self.train_fname = corpus_params.train_url_word
            self.val_fname = corpus_params.val_url_word
            self.test_fname = corpus_params.test_url_word

        self.corpus_open_encoding = corpus_params.open_file_encoding
        self.embedding_open_encoding = embedding_params.open_file_encoding
        self.general_save_encoding = base_params.GENERAL_SAVE_ENCODING

        run_which_model = model_name_full_abbr[self.__class__.__name__]
        corpus_name = corpus_name_full_abbr[corpus_params.__class__.__name__]
        embedding_name = embedding_name_full_abbr.get(embedding_params.__class__.__name__)
        setup_time = tools.get_current_time()
        self.this_model_save_dir = \
            base_params.RESULT_SAVE_DIR + os.path.sep + \
            run_which_model + '_' + corpus_name + '_' + str(embedding_name) + '_' + setup_time
        if not os.path.exists(self.this_model_save_dir):
            os.makedirs(self.this_model_save_dir)

        self.hyperparams = hyperparams
        self.corpus_params = corpus_params
        self.embedding_params = embedding_params

        if self.hyperparams.len_mode == available_len_mode[0]:
            self.input_len = hyperparams.input_len
            self.output_len = hyperparams.output_len
        elif self.hyperparams.len_mode == available_len_mode[1]:
            src_tgt_max_len = reader.get_max_len(self.processed_url, self.corpus_open_encoding)
            self.input_len, self.output_len = src_tgt_max_len[0], src_tgt_max_len[1] - 1
        else:
            self.input_len = None
            self.output_len = None

        # In `word_index` dict: `unk_tag` => 1; word counted max => 2
        # Just take the first `keep_token_num` words in `word_index` into account.
        # Also include unk tag.
        self.keep_token_num = hyperparams.keep_token_num
        self.token_vec_dim = hyperparams.token_vec_dim \
            if embedding_params.vec_dim is None else embedding_params.vec_dim
        self.batch_size = hyperparams.batch_size
        # Set `num_words` to be one more than intended due to a bug in tokenizer of Keras.
        self.tokenizer = reader.fit_tokenizer(self.processed_url, self.keep_token_num+1,
                                              corpus_params.filters, embedding_params.unk_tag,
                                              self.corpus_open_encoding)
        # Actually word_index dict of tokenizer instance includes all words.
        # Get done => Get the right vocab_size.
        self.vocab_size = min(len(self.tokenizer.word_index), self.keep_token_num)
        # May since keras 2.2.4
        self.id2token = self.tokenizer.index_word

        self.pad = self.hyperparams.pad
        self.cut = self.hyperparams.cut

        self.total_samples_count = reader.count_lines(self.processed_url, self.corpus_open_encoding)
        self.train_samples_count = reader.count_lines(self.train_fname, self.corpus_open_encoding)
        self.val_samples_count = reader.count_lines(self.val_fname, self.corpus_open_encoding)
        self.test_samples_count = reader.count_lines(self.test_fname, self.corpus_open_encoding)

        record_info = list()
        record_info.append('\n================ In setup ================\n')
        record_info.append('Vocab size: %d\n' % self.vocab_size)
        record_info.append('Total samples count: %d\n' % self.total_samples_count)
        record_info.append('Train samples count: %d\n' % self.train_samples_count)
        record_info.append('Val samples count: %d\n' % self.val_samples_count)
        record_info.append('Test samples count: %d\n' % self.test_samples_count)
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + base_params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def _do_build(self, x_in_vec_seq, x_in_id_seq, x_mask,
                  y_in_vec_seq, y_in_id_seq, y_mask):
        raise NotImplementedError()

    def build(self):
        """
        define model
        template method pattern
        :return: Model object using the functional API
        """
        # Get done => Not specify the input length of input layer.
        x_in_id_seq = Input(name='x_in', shape=(None,), dtype='int32')
        y_in_id_seq = Input(name='y_in', shape=(None,), dtype='int32')
        if self.pretrained_embeddings_fname:
            token2vec = reader.load_pretrained_token_vecs(self.pretrained_embeddings_fname,
                                                          self.token_vec_dim,
                                                          self.embedding_open_encoding)
            self.embedding_matrix = reader.get_embedding_matrix(token2id=self.tokenizer.word_index,
                                                                token2vec=token2vec,
                                                                vocab_size=self.vocab_size,
                                                                vec_dim=self.token_vec_dim)
            # Get done => The shapes of the embedding matrix and the Embedding layer should match.
            # Get done => Not specify the input len of embedding layer.
            # Set `trainable=True` due to word vec of unk symbol cannot be found.
            # The word vectors will get fine tuned for the specific NLP task during training.
            embedding = Embedding(input_dim=self.vocab_size + 1,
                                  output_dim=self.token_vec_dim,
                                  weights=[self.embedding_matrix],
                                  input_length=None,
                                  name='pretrained_embedding',
                                  trainable=True)
        else:
            embedding = Embedding(input_dim=self.vocab_size + 1,
                                  output_dim=self.token_vec_dim,
                                  name='trainable_embedding')

        x_in_vec_seq = embedding(x_in_id_seq)
        y_in_vec_seq = embedding(y_in_id_seq)
        # print(embedding.input_shape)
        get_pad_mask = GetPadMask()
        x_mask = get_pad_mask(x_in_id_seq)
        y_mask = get_pad_mask(y_in_id_seq)

        attn_hidden_seq = self._do_build(x_in_vec_seq, x_in_id_seq, x_mask,
                                         y_in_vec_seq, y_in_id_seq, y_mask)
        # Apply the same Dense layer instance using same weights to each timestep of input.
        preds = TimeDistributed(Dense(self.vocab_size+1, activation='softmax', name="output_layer",
                                      kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                                      bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                                      activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                                      )
                                )(attn_hidden_seq)

        # end_id = self.tokenizer.word_index[self.embedding_params.end_tag]
        # end_id = tf.Variable([end_id], trainable=False)
        # end_id = K.repeat_elements(end_id, self.batch_size, axis=0)
        # end_id = K.reshape(end_id, [self.batch_size, 1])
        # y_in_id_seq = K.reshape(y_in_id_seq, [self.batch_size, -1])
        # target = concatenate([y_in_id_seq[:, 1:], end_id], axis=-1)

        self.model = Model(inputs=[x_in_id_seq, y_in_id_seq], outputs=preds)

        record_info = list()
        record_info.append('\n================ In build ================\n')
        record_info.append(self.hyperparams.__str__())
        record_info.append(self.corpus_params.__str__())
        record_info.append(str(self.embedding_params))
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + base_params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)
        print('\n############### Model summary ##################')
        self.model.summary()

        return self.model

    def _masked_loss(self, target, preds):
        y_mask = GetPadMask(self.batch_size)(target)
        cross_entropy = K.categorical_crossentropy(target, preds)
        assert K.ndim(cross_entropy) == 2
        loss = K.sum(cross_entropy * y_mask, axis=1, keepdims=True) / K.sum(y_mask, axis=1, keepdims=True)
        return K.reshape(loss, [self.batch_size, -1])

    # Get done => masked acc
    def _masked_categorical_accuracy(self, target, preds):
        y_mask = GetPadMask(self.batch_size)(target)
        raw_tag = K.cast(K.equal(K.argmax(target, axis=-1),
                                 K.argmax(preds, axis=-1)),
                         K.floatx())
        assert K.ndim(raw_tag) == 2
        return raw_tag * y_mask

    # TODO 优化算法
    # 动态学习率 => done，在回调中更改学习率
    # Get done => Masked loss function.
    def compile(self):
        self.model.compile(loss=self._masked_loss,
                           optimizer=self.hyperparams.optimizer,
                           metrics=['accuracy', self._masked_categorical_accuracy])

        # Transformer-based model的图太复杂太乱，没有看的必要
        # 不要在IDE中打开，否则会直接OOM
        # model_vis_url = self.this_model_save_dir + os.path.sep + params.MODEL_VIS_FNAME
        # plot_model(self.model, to_file=model_vis_url, show_shapes=True, show_layer_names=True)

    def fit_generator(self, observe=False, error_text='',
                      beam_width=3, beamsearch_interval=10, is_latin=False):
        train_start = float(time.time())
        early_stopping = EarlyStopping(monitor=self.hyperparams.early_stop_monitor,
                                       patience=self.hyperparams.early_stop_patience,
                                       min_delta=self.hyperparams.early_stop_min_delta,
                                       mode=self.hyperparams.early_stop_mode,
                                       verbose=1)
        # callback_instance.set_model(self.model) => set_model方法由Keras调用
        lr_scheduler = self.hyperparams.lr_scheduler
        save_url = \
            self.this_model_save_dir + os.path.sep + \
            'epoch_{epoch:04d}-{'+self.hyperparams.early_stop_monitor+':.5f}' + '.h5'
        model_saver = ModelCheckpoint(save_url,
                                      monitor=self.hyperparams.early_stop_monitor,
                                      mode=self.hyperparams.early_stop_mode,
                                      save_best_only=True, save_weights_only=True, verbose=1)
        observer = Observer(use_beamsearch=observe, custom_model=self, error_text=error_text,
                            beam_width=beam_width, beamsearch_interval=beamsearch_interval,
                            is_latin=is_latin)
        history = self.model.fit_generator(reader.generate_batch_data_file(self.train_fname,
                                                                           self.tokenizer,
                                                                           self.input_len,
                                                                           self.output_len,
                                                                           self.batch_size,
                                                                           self.vocab_size,
                                                                           self.pad, self.cut,
                                                                           self.corpus_open_encoding),
                                           validation_data=reader.generate_batch_data_file(self.val_fname,
                                                                                           self.tokenizer,
                                                                                           self.input_len,
                                                                                           self.output_len,
                                                                                           self.batch_size,
                                                                                           self.vocab_size,
                                                                                           self.pad, self.cut,
                                                                                           self.corpus_open_encoding),
                                           validation_steps=self.val_samples_count / self.batch_size,
                                           steps_per_epoch=self.train_samples_count / self.batch_size,
                                           epochs=self.hyperparams.train_epoch_times, verbose=1,
                                           callbacks=[model_saver, lr_scheduler, early_stopping, observer])
        tools.show_save_record(self.this_model_save_dir, history, train_start)

    # TODO 评价指标
    def evaluate_generator(self):
        scores = self.model.evaluate_generator(generator=reader.generate_batch_data_file(self.test_fname,
                                                                                         self.tokenizer,
                                                                                         self.input_len,
                                                                                         self.output_len,
                                                                                         self.batch_size,
                                                                                         self.vocab_size,
                                                                                         self.pad, self.cut,
                                                                                         self.corpus_open_encoding),
                                               steps=self.test_samples_count / self.batch_size,
                                               verbose=1)
        record_info = list()
        record_info.append("\n================== 性能评估 ==================\n")
        record_info.append("%s: %.4f\n" % (self.model.metrics_names[0], scores[0]))
        record_info.append("%s: %.2f%%\n" % (self.model.metrics_names[1], scores[1] * 100))
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + base_params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def save(self, model_url):
        self.model.save_weights(model_url)
        print("\n================== 保存模型 ==================")
        print('The weights of', self.__class__.__name__, 'has been saved in', model_url)

    def load(self, model_url):
        self.model.load_weights(model_url, by_name=True)
        print("\n================== 加载模型 ==================")
        print('Model\'s weights have been loaded from', model_url)

    def __call__(self, x, topk, is_latin):
        return beam_search.beam_search(self, x, topk, is_latin)


class Observer(Callback):
    def __init__(self, use_beamsearch, custom_model, error_text,
                 beam_width=3, beamsearch_interval=10, is_latin=False):
        super(Observer, self).__init__()
        self.use_beamsearch = use_beamsearch
        self.custom_model = custom_model
        self.error_text = error_text
        self.beam_width = beam_width
        self.beamsearch_interval = beamsearch_interval
        self.is_latin = is_latin

    def on_epoch_end(self, epoch, logs=None):
        # Get done => Call beam search in callback to observe the process of
        # improvement of proofreading quality.
        # Get done => Call beam search once every specified epochs.
        # Get done => Save beam search result to record file.
        if self.use_beamsearch:
            if epoch % self.beamsearch_interval == 0:
                best_output = beam_search.beam_search(self.custom_model, self.error_text,
                                                      self.beam_width, self.is_latin)
                record_info = list()
                record_info.append('\n======== Beam search when epoch '+str(epoch+1)+' end ========\n')
                record_info.append(self.error_text + ' => ' + best_output + '\n')
                record_str = ''.join(record_info)
                record_url = os.path.join(self.custom_model.this_model_save_dir,
                                          base_params.TRAIN_RECORD_FNAME)
                tools.print_save_str(record_str, record_url)

    def on_train_end(self, logs=None):
        if self.use_beamsearch:
            best_output = beam_search.beam_search(self.custom_model, self.error_text,
                                                  self.beam_width, self.is_latin)
            record_info = list()
            record_info.append('\n======== Beam search when train end ========\n')
            record_info.append(self.error_text + ' => ' + best_output + '\n')
            record_str = ''.join(record_info)
            record_url = os.path.join(self.custom_model.this_model_save_dir,
                                      base_params.TRAIN_RECORD_FNAME)
            tools.print_save_str(record_str, record_url)
