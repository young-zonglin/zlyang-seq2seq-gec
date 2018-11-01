import os
import sys

import jieba

from configs import EmbeddingParams
from configs import NLPCC2018GEC
from utils import split_train_val_test
from utils import tools


# Get done => Whether to cut sentence into character sequence according to
# `char_level` argument in instance of embedding parameters.
# Get done => Add start tag and end tag for source and target sentence in train parallel corpus.
def transform_raw2processed(corpus_params, embedding_params):
    raw_url = corpus_params.raw_url
    open_encoding = corpus_params.open_file_encoding
    save_encoding = corpus_params.save_file_encoding
    char_level = embedding_params.char_level

    if char_level:
        processed_url = corpus_params.processed_url_char
    else:
        processed_url = corpus_params.processed_url_word
    processed_dir = os.path.dirname(processed_url)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    current_func_name = sys._getframe().f_code.co_name
    if raw_url == processed_url:
        print('\n======== In', current_func_name, '========')
        print('Raw url and processed url are the same.')
        print('No split.')
        return
    with open(raw_url, 'r', encoding=open_encoding) as raw_file, \
            open(processed_url, 'w', encoding=save_encoding) as processed_file:
        line_count = 0
        skip_count = 0
        for line in raw_file:
            line_count += 1
            if line_count % 10000 == 0:
                print(line_count, 'lines have been processed.')
            line = line.replace('\n', '').strip()
            filed_list = list(map(str.strip, line.split('\t')))
            num_correct = int(filed_list[1])
            if num_correct+3 != len(filed_list):
                continue
            orig_sen = filed_list[2]
            if char_level:
                orig_sen_segmented = tools.sen2chars(orig_sen, corpus_params.is_latin)
            else:
                orig_sen_segmented = jieba.lcut(orig_sen)
            for i in range(num_correct):
                try:
                    tgt_sen = filed_list[3 + i]
                except IndexError as error:
                    skip_count += 1
                    print(error.__str__())
                    for filed in filed_list:
                        print(filed, end='\t')
                    print()
                    continue
                if char_level:
                    tgt_sen_segmented = tools.sen2chars(tgt_sen, corpus_params.is_latin)
                else:
                    tgt_sen_segmented = jieba.lcut(tgt_sen)
                processed_file.write(embedding_params.start_tag + ' ')
                for token in orig_sen_segmented:
                    processed_file.write(token + ' ')
                processed_file.write(embedding_params.end_tag + ' ')
                processed_file.write('\t')
                processed_file.write(embedding_params.start_tag + ' ')
                for token in tgt_sen_segmented:
                    processed_file.write(token + ' ')
                processed_file.write(embedding_params.end_tag + ' ')
                processed_file.write('\n')
        print('=================================================')
        print(line_count, 'lines have been processed finally.')
        print('skip', skip_count, 'times due to exception.')


def transform_processed2split(corpus_params, embedding_params, force_todo):
    char_level = embedding_params.char_level
    open_encoding = corpus_params.open_file_encoding
    save_encoding = corpus_params.save_file_encoding
    if char_level:
        processed_url = corpus_params.processed_url_char
        train_url = corpus_params.train_url_char
        val_url = corpus_params.val_url_char
        test_url = corpus_params.test_url_char
    else:
        processed_url = corpus_params.processed_url_word
        train_url = corpus_params.train_url_word
        val_url = corpus_params.val_url_word
        test_url = corpus_params.test_url_word
    split_train_val_test(processed_url, train_url, val_url, test_url, force_todo,
                         open_encoding, save_encoding)


# TODO coarse-grained segmentation
def preprocess(corpus_params, embedding_params,
               do_process=True, do_split=True, force_todo=True):
    """
    Pre-process NLPCC 2018 GEC corpus.
    :param corpus_params:
    :param embedding_params:
    :param do_process: Whether to transform raw file to processed file.
    :param do_split: Whether to split processed file into train, val and test data.
    :param force_todo: Whether it is forced to do split.
    :return: Nothing to return.
    """
    if do_process:
        transform_raw2processed(corpus_params, embedding_params)
    if do_split:
        transform_processed2split(corpus_params, embedding_params, force_todo)


if __name__ == '__main__':
    preprocess(NLPCC2018GEC(), EmbeddingParams(),
               do_process=True, do_split=True, force_todo=True)
