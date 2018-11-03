import sys

from configs import NLPCC2018GEC
from configs import TencentAIZhEmbeddings
from utils import get_needed_vectors
from utils import tools


def preprocess(corpus_params, embedding_params):
    """
    Get done => Pre-processing Tencent AI Lab Chinese embeddings.
    :param corpus_params:
    :param embedding_params:
    :return: dict, {token: str => vector: float list}
    """
    return get_needed_vectors(corpus_params, embedding_params)


if __name__ == '__main__':
    token2vec = preprocess(NLPCC2018GEC(), TencentAIZhEmbeddings())
    object_size = tools.byte_to_gb(sys.getsizeof(token2vec))
    print('word2vec dict memory size: %.6f GB.' % object_size)
    print('needed words count:', len(token2vec))
