from configs import available_models, available_corpus, available_embeddings
from configs import get_corpus_params, get_hyperparams, get_embedding_params
from models import ModelFactory
from models.beam_search import beam_search


# TODO structure at test
def apply():
    model_name = available_models[0]
    seq2seq_model = ModelFactory.make_model(model_name)
    hyperparams = get_hyperparams(model_name)
    corpus_name = available_corpus[1]
    corpus_params = get_corpus_params(corpus_name)
    embedding_name = available_embeddings[1]
    embedding_params = get_embedding_params(embedding_name)
    model_url = 'result/AS2SModel_nlpcc-2018-gec_zh-tencent-ailab_2018-11-05 23_07_04/epoch_0002-2.12083.h5'

    seq2seq_model.setup(hyperparams, corpus_params, embedding_params)
    seq2seq_model.build()
    seq2seq_model.load(model_url)
    seq2seq_model.compile()
    seq2seq_model.evaluate_generator()

    error_text = '我在家里一个人学习中文。'
    correct_text = beam_search(custom_model=seq2seq_model, error_text=error_text,
                               beam_width=3, is_latin=False)
    print('=======================================')
    print('error text:', error_text)
    print('correct text:', correct_text)


if __name__ == '__main__':
    apply()
