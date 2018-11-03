import sys
import unittest

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from configs import available_corpus
from configs import get_corpus_params
from utils import reader


class ReaderTest(unittest.TestCase):
    corpus_name = available_corpus[0]
    corpus_params = get_corpus_params(corpus_name)

    @staticmethod
    def test_count_lines():
        url = ReaderTest.corpus_params.raw_url
        line_count = reader.count_lines(url)
        print('======== In', sys._getframe().f_code.co_name + '() function ========')
        print('line count:', line_count)

    @staticmethod
    def test_get_max_len():
        url = ReaderTest.corpus_params.processed_url_word
        res = reader.get_max_len(url)
        print('======== In', sys._getframe().f_code.co_name + '() function ========')
        print(res)

    @staticmethod
    def test_tokenizer():
        texts = ['a a a a', 'b b b', 'c c', 'd']
        tokenizer = Tokenizer(oov_token='<unk>', num_words=6)
        tokenizer.fit_on_texts(texts)
        print('======== In', sys._getframe().f_code.co_name + '() function ========')
        print(tokenizer.word_index)
        print(tokenizer.index_word)
        encodeds = tokenizer.texts_to_sequences(texts)
        print(encodeds)

    @staticmethod
    def test_keras_preprocessing():
        print('======== In', sys._getframe().f_code.co_name + '() function ========')
        print(to_categorical([0, 1, 2, 4], 7))

    @staticmethod
    def test_fit_tokenizer():
        processed_url = ReaderTest.corpus_params.processed_url_word
        tokenizer = reader.fit_tokenizer(processed_url, None)
        print('======== In', sys._getframe().f_code.co_name + '() function ========')
        windexs = list(tokenizer.word_index.items())
        windexs.sort(key=lambda x: x[1])
        print(windexs)


if __name__ == '__main__':
    # method one
    suite = unittest.TestSuite()
    suite.addTest(ReaderTest('test_get_max_len'))

    # method two
    # unittest.main()

    # method three
    # suite = unittest.TestLoader().loadTestsFromTestCase(ReaderTest)

    # and so on
    # suite = unittest.TestLoader().loadTestsFromModule()
    # suite = unittest.TestLoader().loadTestsFromName()

    unittest.TextTestRunner(verbosity=2).run(suite)
