import jieba

from configs import EmbeddingParams
from configs import NLPCC2018GEC
from utils import tools


# Get done => Whether to cut sentence into character sequence according to
# `char_level` argument in instance of embedding parameters.
# Get done => Add start tag and end tag for source and target sentence in train parallel corpus.
def preprocess(raw_url, processed_url, embedding_params,
               open_encoding='utf-8', save_encoding='utf-8'):
    char_level = embedding_params.char_level
    with open(raw_url, 'r', encoding=open_encoding) as raw_file, \
            open(processed_url, 'w', encoding=save_encoding) as processed_file:
        line_count = 0
        skip_count = 0
        for line in raw_file:
            line_count += 1
            if line_count % 10000 == 0:
                print(line_count, 'lines have been processed.')
            line = line.strip().replace('\n', '')
            filed_list = line.split('\t')
            num_correct = int(filed_list[1])
            if num_correct+3 != len(filed_list):
                continue
            orig_sen = filed_list[2]
            if char_level:
                orig_sen_segmented = tools.sen2chars(orig_sen)
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
                    tgt_sen_segmented = tools.sen2chars(tgt_sen)
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


if __name__ == '__main__':
    nlpcc_2018_gec = NLPCC2018GEC()
    preprocess(nlpcc_2018_gec.raw_url, nlpcc_2018_gec.processed_url, EmbeddingParams())
