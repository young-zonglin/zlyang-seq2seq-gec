import os
import sys

from configs import base_params


class CorpusParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        self.open_file_encoding = 'utf-8'
        self.save_file_encoding = 'utf-8'
        self.is_latin = False

        self.corpus_root = os.path.join(base_params.PROJECT_ROOT, 'data', 'parallel-corpus')
        self.raw_url = None

        self.processed_url_word = None
        self.train_url_word = None
        self.val_url_word = None
        self.test_url_word = None

        self.processed_url_char = None
        self.train_url_char = None
        self.val_url_char = None
        self.test_url_char = None

        self.filters = ''

    def __str__(self):
        ret_info = list()
        ret_info.append("open file encoding: " + self.open_file_encoding + '\n')
        ret_info.append("save file encoding: " + self.save_file_encoding + '\n')
        ret_info.append("is latin language: " + str(self.is_latin) + '\n\n')

        ret_info.append("raw url: " + str(self.raw_url) + '\n\n')

        ret_info.append("word level processed url: " + str(self.processed_url_word) + '\n')
        ret_info.append("word level train url: " + str(self.train_url_word) + '\n')
        ret_info.append("word level val url: " + str(self.val_url_word) + '\n')
        ret_info.append("word level test url: " + str(self.test_url_word) + '\n\n')

        ret_info.append("char level processed url: " + str(self.processed_url_char) + '\n')
        ret_info.append("char level train url: " + str(self.train_url_char) + '\n')
        ret_info.append("char level val url: " + str(self.val_url_char) + '\n')
        ret_info.append("char level test url: " + str(self.test_url_char) + '\n\n')

        ret_info.append("filters: " + self.filters + '\n\n')

        return ''.join(ret_info)


class JustForTest(CorpusParams):
    def __init__(self):
        super(JustForTest, self).__init__()

        # just for test
        # train, val and test data are the same.
        just_for_test = os.path.join(self.corpus_root, 'just_for_test')

        self.raw_url = just_for_test
        self.processed_url_word = just_for_test
        self.train_url_word = just_for_test
        self.val_url_word = just_for_test
        self.test_url_word = just_for_test

    def __str__(self):
        ret_info = list()
        ret_info.append('================== ' + self.current_classname + ' ==================\n')
        super_str = super(JustForTest, self).__str__()
        return ''.join(ret_info) + super_str


class NLPCC2018GEC(CorpusParams):
    def __init__(self):
        super(NLPCC2018GEC, self).__init__()

        nlpcc_2018_gec_dir = os.path.join(self.corpus_root, 'NLPCC-2018-GEC')
        self.raw_data_dir = os.path.join(nlpcc_2018_gec_dir, 'raw_data')
        self.processed_data_dir = os.path.join(nlpcc_2018_gec_dir, 'processed_data')

        # raw, processed, train, val, test
        self.raw_url = os.path.join(self.raw_data_dir, 'data.train')

        self.processed_url_word = os.path.join(self.processed_data_dir, 'data.processed.word')
        self.train_url_word = os.path.join(self.processed_data_dir, 'gec_train.word')
        self.val_url_word = os.path.join(self.processed_data_dir, 'gec_val.word')
        self.test_url_word = os.path.join(self.processed_data_dir, 'gec_test.word')

        self.processed_url_char = os.path.join(self.processed_data_dir, 'data.processed.char')
        self.train_url_char = os.path.join(self.processed_data_dir, 'gec_train.char')
        self.val_url_char = os.path.join(self.processed_data_dir, 'gec_val.char')
        self.test_url_char = os.path.join(self.processed_data_dir, 'gec_test.char')

    def __str__(self):
        ret_info = list()
        ret_info.append('================== '+self.current_classname+' ==================\n')
        ret_info.append("raw data dir: " + self.raw_data_dir + '\n')
        ret_info.append("processed data dir: " + self.processed_data_dir + '\n\n')

        super_str = super(NLPCC2018GEC, self).__str__()
        return ''.join(ret_info) + super_str


corpus_name_abbr_full = {'just-for-test': JustForTest().__class__.__name__,
                         'nlpcc-2018-gec': NLPCC2018GEC().__class__.__name__}
corpus_name_full_abbr = {v: k for k, v in corpus_name_abbr_full.items()}
available_corpus = ['just-for-test', 'nlpcc-2018-gec']


def get_corpus_params(corpus_name):
    if corpus_name == available_corpus[0]:
        return JustForTest()
    elif corpus_name == available_corpus[1]:
        return NLPCC2018GEC()
    else:
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() func, corpus_name value error.')


if __name__ == '__main__':
    print(CorpusParams())
    print(JustForTest())
    print(NLPCC2018GEC())
