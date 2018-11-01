import sys

import jieba
import numpy as np

from utils import tools


# Get done => beam search
# TODO <start> <end> <unk>
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
        error_text = [token for token in jieba.lcut(error_text) if not token.isspace()]
    error_text = ' '.join(error_text)
    x_in_id_seq = custom_model.tokenizer.texts_to_sequences([error_text])[0]
    assert type(x_in_id_seq) is list
    assert type(x_in_id_seq[0]) is int
    start_id = custom_model.tokenizer.word_index[custom_model.embedding_params.start_tag]
    end_id = custom_model.tokenizer.word_index[custom_model.embedding_params.end_tag]
    res2ppl = {' '.join([str(start_id)]): 0}
    output_len = len(x_in_id_seq) if custom_model.output_len is None else custom_model.output_len
    x_in_id_seq = np.array([x_in_id_seq], dtype=np.int32)
    assert x_in_id_seq.ndim == 2

    for i in range(output_len):
        _res2ppl = {}  # Save result => ppl temporarily
        for res in res2ppl.keys():
            res = list(map(int, res.split(' ')))
            if res[-1] == end_id:
                continue

            res = np.array([res], dtype=np.int32)
            proba = custom_model.model.predict([x_in_id_seq, res])
            assert type(proba) is np.ndarray
            # get the proba distribution of next token.
            proba = np.reshape(proba[:, -1, :], (custom_model.vocab_size+1,))
            assert proba.shape == (custom_model.vocab_size+1,)

            log_proba = np.log(proba + 1e-6)
            index_topk = np.argsort(-log_proba, axis=-1)[:beam_width]
            res = [str(token_id) for token_id in res[0]]
            for top_i in index_topk:
                assert type(top_i) is np.int64
                _res2ppl[' '.join(res+[str(top_i)])] = res2ppl[' '.join(res)]+log_proba[top_i]

        # All results end with <end>.
        if not _res2ppl:
            break
        else:
            res2ppl = _res2ppl

    # Normalize the ppl.
    for sen, ppl in res2ppl.items():
        res2ppl[sen] = ppl / len(sen.split(' '))
    # pick up the best sentence
    max_sen = max(res2ppl, key=res2ppl.get)
    return tools.ids2seq(max_sen, custom_model.id2token, custom_model.embedding_params, return_sen=True)
