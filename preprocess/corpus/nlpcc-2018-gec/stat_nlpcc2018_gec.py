from configs import available_corpus
from configs import get_corpus_params
from utils import reader


def stat(wcounts):
    times_stat = {}
    for word, count in wcounts:
        if times_stat.get(count) is None:
            times_stat[count] = 1
        else:
            times_stat[count] += 1
    all_words_num = len(wcounts)
    assert all_words_num == sum(map(lambda x: x[1], list(times_stat.items())))
    times_stat = list(times_stat.items())
    times_stat.sort(key=lambda x: x[1], reverse=True)
    for times, cnt in times_stat:
        print(cnt, 'words occur', times, 'times, the ratio is', cnt / all_words_num * 100, '%')


if __name__ == '__main__':
    nlpcc2018_gec = available_corpus[1]
    corpus_params = get_corpus_params(nlpcc2018_gec)
    processed_url = corpus_params.processed_url_word
    corpus_open_encoding = corpus_params.open_file_encoding

    tokenizer = reader.fit_tokenizer(processed_url, None)
    wcounts = list(tokenizer.word_counts.items())

    all_words = reader.read_tokens(processed_url, corpus_open_encoding)

    print('all words num come from tokenizer:', len(wcounts))
    print('all words num come from `read_tokens` function:', len(all_words))
    if len(wcounts) == len(all_words):
        print('ok, they are the same.')

    stat(wcounts)
