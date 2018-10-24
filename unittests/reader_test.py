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


if __name__ == '__main__':
    unittest.main()
