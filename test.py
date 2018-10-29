from configs import available_models, available_corpus, available_embeddings
from configs import get_corpus_params, get_hyperparams, get_embedding_params
from models import ModelFactory
from utils import tools


def main():
    run_this_model = available_models[0]
    seq2seq_model = ModelFactory.make_model(run_this_model)
    hyperparams = get_hyperparams(run_this_model)
    hyperparams.batch_size = 1
    corpus_name = available_corpus[0]
    corpus_params = get_corpus_params(corpus_name)
    embedding_name = available_embeddings[0]
    embedding_params = get_embedding_params(embedding_name)
    embedding_params = None

    error_text = '我在家里一个人学习中文。'
    tools.train_model(seq2seq_model, hyperparams, corpus_params, embedding_params,
                      observe=True, error_text=error_text,
                      beam_width=3, beamsearch_interval=10)


if __name__ == '__main__':
    main()
