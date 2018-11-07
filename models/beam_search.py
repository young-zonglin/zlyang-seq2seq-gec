import sys

import jieba
import numpy as np

from utils import tools


# Get done => beam search
# TODO recover from unk symbol.
def beam_search(custom_model, error_text, beam_width=3, is_latin=False):
    """
    Beam search decoding to generate correct text.
    :param custom_model:
    :param error_text: str, like "今天天气很耗"
    :param beam_width:
    :param is_latin:
    :return: correct text.
    """
    if error_text.__class__ != str:
        raise TypeError('In ' + sys._getframe().f_code.co_name +
                        '() function, error_text should be a str.')

    beam_width = custom_model.vocab_size if beam_width > custom_model.vocab_size else beam_width
    if custom_model.embedding_params.char_level:
        error_text = tools.sen2chars(error_text, is_latin)
    else:
        error_text = [word for word in jieba.lcut(error_text) if not word.isspace()]
    error_text = ' '.join(error_text)
    x_in_id_seq = custom_model.tokenizer.texts_to_sequences([error_text])[0]
    assert type(x_in_id_seq) is list
    assert type(x_in_id_seq[0]) is int
    start_id = custom_model.tokenizer.word_index[custom_model.embedding_params.start_tag]
    end_id = custom_model.tokenizer.word_index[custom_model.embedding_params.end_tag]
    results = [[start_id]]
    ppls = [0]
    output_len = len(x_in_id_seq) if custom_model.output_len is None else custom_model.output_len
    x_in_id_seq = np.array([x_in_id_seq], dtype=np.int32)
    assert x_in_id_seq.ndim == 2

    for i in range(output_len):
        _res = []  # Save result temporarily.
        _ppls = []  # Save ppl temporarily.
        for j in range(len(results)):
            res = results[j]
            if res[-1] == end_id and i != 0:  # Be careful when `start_tag` == `end_tag`
                continue

            proba = custom_model.model.predict([x_in_id_seq, np.array([res], dtype=np.int32)])
            assert type(proba) is np.ndarray
            # get the proba distribution of next token.
            proba = np.reshape(proba[:, -1, :], (custom_model.vocab_size+1,))
            assert proba.shape == (custom_model.vocab_size+1,)
            log_proba = np.log(proba + 1e-6)
            for index in range(log_proba.shape[0]):
                assert type(index) is int
                _res.append(res + [index])
                _ppls.append(ppls[j]+log_proba[index])

        # All results end with `end_tag`.
        if not _res or not _ppls:
            break
        else:
            # Normalize the ppl.
            for m in range(len(_res)):
                _ppls[m] = _ppls[m] / len(_res[m])
            index_topk = np.argsort(-np.array(_ppls), axis=-1)[:beam_width]
            results = []
            ppls = []
            for index in index_topk:
                results.append(_res[index])
                ppls.append(_ppls[index])

    # pick up the best sentence
    max_index = np.argmax(ppls)
    assert type(max_index) is np.int64
    max_sen = results[max_index]
    return tools.ids2seq(max_sen, custom_model.id2token, custom_model.embedding_params, return_sen=True)
