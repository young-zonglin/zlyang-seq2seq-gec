from configs import available_corpus
from configs import get_corpus_params
from utils import reader


def stat(tcounts):
    times_stat = {}
    for token, count in tcounts:
        if times_stat.get(count) is None:
            times_stat[count] = 1
        else:
            times_stat[count] += 1
    vocab_size = len(tcounts)
    assert vocab_size == sum(map(lambda x: x[1], list(times_stat.items())))
    all_tokens_num = sum(map(lambda x: x[1], tcounts))
    print('all tokens num:', all_tokens_num)
    times_stat = list(times_stat.items())
    times_stat.sort(key=lambda x: x[1], reverse=True)
    for times, cnt in times_stat:
        print('{0:6} words occur {1:10} times, the ratio of vocab_size is {2:>10.6f}% '
              '| the ratio of all tokens num is {3:>10.6f}%'
              .format(cnt, times, cnt / vocab_size * 100, cnt*times / all_tokens_num * 100))


if __name__ == '__main__':
    nlpcc2018_gec = available_corpus[1]
    corpus_params = get_corpus_params(nlpcc2018_gec)
    processed_url = corpus_params.processed_url_word
    corpus_open_encoding = corpus_params.open_file_encoding

    tokenizer = reader.fit_tokenizer(processed_url, None)
    tcounts = list(tokenizer.word_counts.items())

    all_tokens = reader.read_tokens(processed_url, corpus_open_encoding)

    print('vocabulary size come from tokenizer:', len(tcounts))
    print('vocab size come from `read_tokens` function:', len(all_tokens))
    if len(tcounts) == len(all_tokens):
        print('ok, they are the same.')

    stat(tcounts)
