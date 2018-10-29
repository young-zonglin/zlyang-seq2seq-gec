import unittest

from configs import available_corpus
from configs import get_corpus_params
from utils import reader


class ReaderTest(unittest.TestCase):
    corpus_name = available_corpus[1]
    corpus_params = get_corpus_params(corpus_name)

    @staticmethod
    def test_count_lines():
        url = ReaderTest.corpus_params.raw_url
        line_count = reader.count_lines(url)
        print('line count', line_count)

    @staticmethod
    def test_get_max_len():
        url = ReaderTest.corpus_params.processed_url
        res = reader.get_max_len(url)
        print(res)


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
