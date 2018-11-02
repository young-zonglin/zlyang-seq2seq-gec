import jieba

from configs import EmbeddingParams
from configs import NLPCC2018GEC
from utils import split_train_val_test
from utils import tools


# Get done => Whether to cut sentence into character sequence according to
# `char_level` argument in instance of embedding parameters.
# Get done => Add start tag and end tag for source and target sentence in train parallel corpus.
def transform_line(line, tgt_file, embedding_params, is_latin):
    char_level = embedding_params.char_level
    line = line.replace('\n', '').strip()
    filed_list = list(map(str.strip, line.split('\t')))
    num_correct = int(filed_list[1])
    if num_correct+3 != len(filed_list):
        return
    orig_sen = filed_list[2]
    if char_level:
        orig_sen_segmented = tools.sen2chars(orig_sen, is_latin)
    else:
        orig_sen_segmented = jieba.lcut(orig_sen)
    for i in range(num_correct):
        try:
            tgt_sen = filed_list[3 + i]
        except IndexError as error:
            print(error.__str__())
            for filed in filed_list:
                print(filed, end='\t')
            print()
            continue
        if char_level:
            tgt_sen_segmented = tools.sen2chars(tgt_sen, is_latin)
        else:
            tgt_sen_segmented = jieba.lcut(tgt_sen)
        tgt_file.write(embedding_params.start_tag + ' ')
        for token in orig_sen_segmented:
            tgt_file.write(token + ' ')
        tgt_file.write(embedding_params.end_tag + ' ')
        tgt_file.write('\t')
        tgt_file.write(embedding_params.start_tag + ' ')
        for token in tgt_sen_segmented:
            tgt_file.write(token + ' ')
        tgt_file.write(embedding_params.end_tag + ' ')
        tgt_file.write('\n')


# TODO coarse-grained segmentation
# Get done => Split firstly and then process line due to
#   a wrong sentence may correspond to multiple correct results.
# TODO Add correct to crt sen pair.
def preprocess(corpus_params, embedding_params, force_todo=True):
    """
    Pre-process NLPCC 2018 GEC corpus.
    :param corpus_params:
    :param embedding_params:
    :param force_todo: Whether it is forced to do split and process.
    :return: Nothing to return.
    """
    split_train_val_test(transform_line, corpus_params, embedding_params, force_todo)


if __name__ == '__main__':
    preprocess(NLPCC2018GEC(), EmbeddingParams(), force_todo=True)
