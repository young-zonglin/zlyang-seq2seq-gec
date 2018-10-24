import io
import os
import re
import sys

import numpy as np
import numpy.random as rdm
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from configs import base_params
from utils import tools

match_newline_pattern = re.compile('\n+')


def load_word_vecs(fname, head_n=None, open_encoding='utf-8'):
    """
    装载前N个词向量
    :param fname:
    :param head_n: head n word vectors will be loaded
    :param open_encoding: open file encoding
    :return: dict, {word: str => vector: float list}
    """
    line_count = 0
    word2vec = {}
    try:
        fin = io.open(fname, 'r', encoding=open_encoding,
                      newline='\n', errors='ignore')
    except FileNotFoundError as error:
        print(error)
        return word2vec

    for line in fin:
        # load head n word vectors
        if head_n and head_n.__class__ == int:
            line_count += 1
            if line_count > head_n:
                break
        tokens = line.strip().replace('\n', '').split(' ')
        # map是一个类，Python中的高阶函数，类似于Scala中的array.map(func)
        # 将传入的函数作用于传入的可迭代对象（例如list）的每一个元素之上
        # float也是一个类
        # Convert a string or number to a floating point number, if possible.
        word2vec[tokens[0]] = list(map(float, tokens[1:]))
    fin.close()
    return word2vec


def generate_text_from_corpus(path, open_encoding='utf-8'):
    """
    生成器函数，一次返回一个文本的全部内容
    :param path: corpus path
    :param open_encoding: open file encoding
    :return: 返回迭代器，可以遍历path下所有文件的内容
    """
    if not os.path.isdir(path):
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         ' func, argument should be path.')
    fnames = tools.get_fnames_under_path(path)
    for fname in fnames:
        with open(fname, 'r', encoding=open_encoding) as file:
            yield file.read()


def read_words(url, open_encoding='utf-8'):
    """
    Read all distinct words.
    :param url:
    :param open_encoding:
    :return: set, {'apple', 'banana', ...}
    """
    ret_words = set()
    if os.path.isdir(url):
        for text in generate_text_from_corpus(url, open_encoding):
            for line in match_newline_pattern.split(text):
                for word in line.split():
                    ret_words.add(word)
    elif os.path.isfile(url):
        with open(url, 'r', encoding=open_encoding) as file:
            for line in file:
                for word in line.split():
                    ret_words.add(word)
    return ret_words


def get_needed_vectors(url, full_vecs_fname, needed_vecs_fname,
                       open_encoding='utf-8', save_encoding='utf-8'):
    """
    1. Read all distinct words from processed train files.
    2. If word not in needed word vectors file, get it's vector from full word vectors file.
    3. Return needed word vectors dict.
    :return: dict, {word: str => vector: float list}
    """
    all_words = read_words(url, open_encoding)
    needed_word2vec = load_word_vecs(needed_vecs_fname, open_encoding=open_encoding)

    is_all_in_needed = True
    for word in all_words:
        if word not in needed_word2vec:
            print(word, 'not in needed word2vec.')
            is_all_in_needed = False
    if not is_all_in_needed:
        with open(full_vecs_fname, 'r', encoding=open_encoding) as full_file, \
                open(needed_vecs_fname, 'a', encoding=save_encoding) as needed_file:
            line_count = 0
            print('============ In ' + sys._getframe().f_code.co_name + '() func ============')
            for line in full_file:
                line_count += 1
                if line_count % 100000 == 0:
                    print(line_count, 'has been processed.')
                tokens = line.strip().split()
                word = tokens[0]
                if word in all_words and word not in needed_word2vec:
                    for token in tokens:
                        needed_file.write(token+' ')
                    needed_file.write('\n')
        needed_word2vec = load_word_vecs(needed_vecs_fname, open_encoding=open_encoding)
    else:
        print('All words in needed word2vec.')
    return needed_word2vec


def split_train_val_test(raw_url, train_fname, val_fname, test_fname,
                         open_encoding='utf-8', save_encoding='utf-8'):
    """
    randomly split raw data corpus to train data, val data and test data.
    train : val : test = 8:1:1
    test data used for unbiased estimation of model performance.
    :return: None
    """
    current_func_name = sys._getframe().f_code.co_name
    if raw_url in [train_fname, val_fname, test_fname]:
        print('\n======== In', current_func_name, '========')
        print('Raw path and train, val, test data filenames are the same.')
        print('No split.')
        return
    if os.path.exists(train_fname) and os.path.exists(val_fname) and os.path.exists(test_fname):
        print('\n======== In', current_func_name, '========')
        print('Train, val and test data already exists.')
        return
    with open(train_fname, 'w', encoding=save_encoding) as train_file, \
            open(val_fname, 'w', encoding=save_encoding) as val_file, \
            open(test_fname, 'w', encoding=save_encoding) as test_file:
        if os.path.isdir(raw_url):
            for text in generate_text_from_corpus(raw_url, open_encoding):
                for line in match_newline_pattern.split(text):
                    if line == '':
                        continue
                    rand_value = rdm.rand()
                    if rand_value >= 0.2:
                        train_file.write(line + '\n')
                    elif 0.1 <= rand_value < 0.2:
                        val_file.write(line + '\n')
                    else:
                        test_file.write(line + '\n')
        elif os.path.isfile(raw_url):
            with open(raw_url, 'r', encoding=open_encoding) as raw_file:
                for line in raw_file:
                    if line == '' or line == '\n':
                        continue
                    rand_value = rdm.rand()
                    if rand_value >= 0.2:
                        train_file.write(line + '\n')
                    elif 0.1 <= rand_value < 0.2:
                        val_file.write(line + '\n')
                    else:
                        test_file.write(line + '\n')


def load_pretrained_word_vecs(fname, open_encoding='utf-8'):
    """
    load needed word vectors
    :return: dict, {word: str => embedding: numpy array}
    """
    word2vec = load_word_vecs(fname, open_encoding=open_encoding)
    for word, embedding in word2vec.items():
        embedding = np.asarray(embedding, dtype=np.float32)
        word2vec[word] = embedding
    return word2vec


def get_embedding_matrix(word2id, word2vec, vec_dim):
    """
    turn word2vec dict to embedding matrix
    :param word2id: dict
    :param word2vec: dict
    :param vec_dim: embedding dim
    :return: embedding matrix
    """
    embedding_matrix = np.zeros((len(word2id)+1, vec_dim))
    for word, index in word2id.items():
        # words not found in word2vec will be all-zeros.
        embedding = word2vec.get(word)
        if embedding is not None:
            embedding_matrix[index] = embedding
    return embedding_matrix


def count_lines(url, open_encoding='utf-8'):
    line_count = 0
    if os.path.isdir(url):
        for text in generate_text_from_corpus(url, open_encoding):
            for line in match_newline_pattern.split(text):
                if line == '':
                    continue
                line_count += 1
    else:
        with open(url, 'r', encoding=open_encoding) as file:
            for _ in file:
                line_count += 1
    return line_count


def fit_tokenizer(raw_url, keep_word_num, filters='', oov_tag='<UNK>',
                  char_level=False, open_encoding='utf-8'):
    """
    use corpus to fit tokenizer.
    :param raw_url: corpus path or filename.
    :param keep_word_num: the maximum number of words to keep, based
            on word frequency. Only the most common `keep_word_num` words
            will be kept.
    :param filters: a string where each element is a character that will be
            filtered from the texts. The default is empty string.
    :param oov_tag: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls.
    :param char_level: if True, every character will be treated as a token.
    :param open_encoding:
    :return: tokenizer fitted by corpus.
    """
    tokenizer = Tokenizer(num_words=keep_word_num,
                          filters=filters,
                          oov_token=oov_tag,
                          char_level=char_level)
    if os.path.isdir(raw_url):
        tokenizer.fit_on_texts(generate_text_from_corpus(raw_url, open_encoding))
    else:
        file = open(raw_url, 'r', encoding=open_encoding)
        text = file.read()
        file.close()
        texts = [text]
        tokenizer.fit_on_texts(texts)
    return tokenizer


def generate_in_out_pair_file(fname, tokenizer, open_encoding='utf-8'):
    """
    generate func, generate a input-output pair at a time.
    yield a tuple at a time.
    will loop indefinitely on the dataset.
    :return: a iterator
    """
    while 1:
        with open(fname, 'r', encoding=open_encoding) as file:
            for line in file:
                if line and line != '\n':
                    in_out_pair = line.split(base_params.SEPARATOR)
                    if len(in_out_pair) != 2:
                        continue
                    source, target = in_out_pair[0], in_out_pair[1]
                    encodeds = tokenizer.texts_to_sequences([source, target])
                    yield encodeds[0], encodeds[1]


def process_format_model_in(in_out_pairs, input_len, output_len, vocab_size,
                            pad='pre', cut='pre'):
    """
    Process the format of the input-output pairs
    to meet the input requirements of the model.
    :param in_out_pairs: [(in, out), ...]
    :param input_len:
    :param output_len:
    :param vocab_size:
    :param pad:
    :param cut:
    :return: ({'x_in': x, 'y_in': y_in}, y_out)
    """
    x = []
    y_in = []
    y_out = []
    for in_out_pair in in_out_pairs:
        x.append(in_out_pair[0])
        y_in.append(in_out_pair[1][:-1])
        y_out.append(in_out_pair[1][1:])

    # list of lists => 2d numpy array
    x = pad_sequences(x, maxlen=input_len, padding=pad, truncating=cut)
    y_in = pad_sequences(y_in, maxlen=output_len, padding=pad, truncating=cut)

    y_out = pad_sequences(y_out, maxlen=output_len, padding=pad, truncating=cut)
    # y.shape == (batch size, output length, one-hot vec dim)
    y_out = np.asarray([to_categorical(y_out[i], vocab_size+1) for i in range(len(y_out))])

    return {'x_in': x, 'y_in': y_in}, y_out


def generate_batch_data_file(fname, tokenizer, input_len, output_len, batch_size,
                             vocab_size, pad, cut, open_encoding):
    """
    Generator function, generating a batch data at a time.
    Will loop indefinitely on the dataset.
    :return: a iterator
    """
    batch_samples_count = 0
    in_out_pairs = list()
    for in_out_pair in generate_in_out_pair_file(fname, tokenizer, open_encoding):
        # Return fixed and the same number of samples each time.
        if batch_samples_count < batch_size - 1:
            in_out_pairs.append(in_out_pair)
            batch_samples_count += 1
        else:
            in_out_pairs.append(in_out_pair)
            x, y = process_format_model_in(in_out_pairs, input_len, output_len, vocab_size, pad, cut)
            yield x, y
            in_out_pairs = list()
            batch_samples_count = 0
