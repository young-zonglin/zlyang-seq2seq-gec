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


def load_vecs(fname, head_n=None, open_encoding='utf-8'):
    """
    装载前N个嵌入向量
    :param fname:
    :param head_n: head n embedding vectors will be loaded
    :param open_encoding: open file encoding
    :return: dict, {token: str => vector: float list}
    """
    line_count = 0
    token2vec = {}
    try:
        fin = io.open(fname, 'r', encoding=open_encoding,
                      newline='\n', errors='ignore')
    except FileNotFoundError as error:
        print(error)
        return token2vec

    first_line = fin.readline()
    tmp = first_line.split()
    words_num, vec_dim = int(tmp[0]), int(tmp[1])
    for line in fin:
        # load head n embedding vectors
        if head_n and head_n.__class__ == int:
            line_count += 1
            if line_count > head_n:
                break
        tokens = line.split()
        if len(tokens) != vec_dim+1:
            continue
        # map是一个类，Python中的高阶函数，类似于Scala中的array.map(func)
        # 将传入的函数作用于传入的可迭代对象（例如list）的每一个元素之上
        # float也是一个类
        # Convert a string or number to a floating point number, if possible.
        token2vec[tokens[0]] = list(map(float, tokens[1:]))
    fin.close()
    return token2vec


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


def read_tokens(url, open_encoding='utf-8'):
    """
    Read all distinct tokens.
    :param url:
    :param open_encoding:
    :return: set, {'apple', 'banana', ...}
    """
    ret_tokens = set()
    if os.path.isdir(url):
        for text in generate_text_from_corpus(url, open_encoding):
            for line in match_newline_pattern.split(text):
                for token in line.split():
                    ret_tokens.add(token)
    elif os.path.isfile(url):
        with open(url, 'r', encoding=open_encoding) as file:
            for line in file:
                for token in line.split():
                    ret_tokens.add(token)
    return ret_tokens


def get_needed_vectors(url, full_vecs_fname, needed_vecs_fname,
                       open_encoding='utf-8', save_encoding='utf-8'):
    """
    1. Read all distinct tokens from processed train files.
    2. If token not in needed token vectors file, get it's vector from full token vectors file.
    3. Return needed token vectors dict.
    :return: dict, {token: str => vector: float list}
    """
    all_tokens = read_tokens(url, open_encoding)
    needed_token2vec = load_vecs(needed_vecs_fname, open_encoding=open_encoding)

    is_all_in_needed = True
    for token in all_tokens:
        if token not in needed_token2vec:
            print(token, 'not in needed token2vec.')
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
                token = tokens[0]
                if token in all_tokens and token not in needed_token2vec:
                    for token in tokens:
                        needed_file.write(token+' ')
                    needed_file.write('\n')
        needed_token2vec = load_vecs(needed_vecs_fname, open_encoding=open_encoding)
    else:
        print('All tokens in needed token2vec.')
    return needed_token2vec


def split_train_val_test(raw_url, train_fname, val_fname, test_fname, force_todo=False,
                         open_encoding='utf-8', save_encoding='utf-8'):
    """
    randomly split raw data corpus to train data, val data and test data.
    train : val : test = 8:1:1
    test data used for unbiased estimation of model performance.
    :return: Nothing to return.
    """
    current_func_name = sys._getframe().f_code.co_name
    if raw_url in [train_fname, val_fname, test_fname]:
        print('\n======== In', current_func_name, '========')
        print('Raw path and train, val, test data filenames are the same.')
        print('No split.')
        return
    if not force_todo and os.path.exists(train_fname) and os.path.exists(val_fname) and os.path.exists(test_fname):
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
                    line = line.strip().replace('\n', '')
                    if line == '':
                        continue
                    rand_value = rdm.rand()
                    if rand_value >= 0.2:
                        train_file.write(line + '\n')
                    elif 0.1 <= rand_value < 0.2:
                        val_file.write(line + '\n')
                    else:
                        test_file.write(line + '\n')


def load_pretrained_token_vecs(fname, open_encoding='utf-8'):
    """
    load needed token vectors
    :return: dict, {token: str => embedding: numpy array}
    """
    token2vec = load_vecs(fname, open_encoding=open_encoding)
    for token, embedding in token2vec.items():
        embedding = np.asarray(embedding, dtype=np.float32)
        token2vec[token] = embedding
    return token2vec


def get_embedding_matrix(token2id, token2vec, vec_dim):
    """
    turn token2vec dict to embedding matrix
    :param token2id: dict
    :param token2vec: dict
    :param vec_dim: embedding dim
    :return: embedding matrix
    """
    embedding_matrix = np.zeros((len(token2id) + 1, vec_dim))
    for token, index in token2id.items():
        # tokens not found in token2vec will be all-zeros.
        embedding = token2vec.get(token)
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
            for line in file:
                if line != '\n' and line != '':
                    line_count += 1
    return line_count


def get_max_len(url, open_encoding='utf-8'):
    src_max_len = -1
    tgt_max_len = -1
    with open(url, 'r', encoding=open_encoding) as file:
        for line in file:
            field_list = line.split('\t')
            if len(field_list) != 2:
                continue
            src, tgt = field_list[0].strip().split(' '), field_list[1]\
                .strip().replace('\n', '').split(' ')
            src_max_len = max(src_max_len, len(src))
            tgt_max_len = max(tgt_max_len, len(tgt))
    return src_max_len, tgt_max_len


def fit_tokenizer(raw_url, keep_token_num, filters='',
                  oov_tag='<unk>', open_encoding='utf-8'):
    """
    use corpus to fit tokenizer.
    :param raw_url: corpus path or filename.
    :param keep_token_num: the maximum number of tokens to keep, based
            on token frequency. Only the most common `keep_token_num` tokens
            will be kept.
    :param filters: a string where each element is a character that will be
            filtered from the texts. The default is empty string.
    :param oov_tag: if given, it will be added to token_index and used to
            replace out-of-vocabulary tokens during text_to_sequence calls.
    :param open_encoding:
    :return: tokenizer fitted by corpus.
    """
    tokenizer = Tokenizer(num_words=keep_token_num,
                          filters=filters,
                          oov_token=oov_tag)
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
                line = line.replace('\n', '').strip()
                if line:
                    in_out_pair = line.split(base_params.SEPARATOR)
                    if len(in_out_pair) != 2:
                        continue
                    source, target = in_out_pair[0].strip(), in_out_pair[1].strip()
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

    if input_len is None or output_len is None:
        input_len = max([len(src) for src in x])
        output_len = max([len(tgt) for tgt in y_out])

    # list of lists => 2d numpy array
    x = pad_sequences(x, maxlen=input_len, padding=pad, truncating=cut)
    y_in = pad_sequences(y_in, maxlen=output_len, padding=pad, truncating=cut)

    y_out = pad_sequences(y_out, maxlen=output_len, padding=pad, truncating=cut)
    # y_out.shape == (batch size, output length, one-hot vec dim)
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
