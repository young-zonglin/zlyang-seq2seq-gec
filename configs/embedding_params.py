import os
import sys

from configs import base_params


class EmbeddingParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        self.char_level = False
        self.vec_dim = None

        self.start_tag = '<start>'
        self.end_tag = '<end>'
        self.unk_tag = '<unk>'

        self.open_file_encoding = 'utf-8'
        self.save_file_encoding = 'utf-8'

        self.pretrained_embeddings_root = os.path.join(base_params.PROJECT_ROOT,
                                                       'data', 'pretrained-embeddings')
        self.raw_pretrained_embeddings_url = None
        self.pretrained_embeddings_url = None

    def __str__(self):
        ret_info = list()
        ret_info.append("char level: " + str(self.char_level) + '\n')
        ret_info.append("dim of token vec: " + str(self.vec_dim) + '\n\n')

        ret_info.append("start tag: " + self.start_tag + '\n')
        ret_info.append("end tag: " + self.end_tag + '\n')
        ret_info.append("unk tag: " + self.unk_tag + '\n\n')

        ret_info.append("open file encoding: " + self.open_file_encoding + '\n')
        ret_info.append("save file encoding: " + self.save_file_encoding + '\n\n')

        ret_info.append('pretrained embeddings root: ' + self.pretrained_embeddings_root + '\n')
        ret_info.append('raw pretrained embeddings url: ' +
                        str(self.raw_pretrained_embeddings_url) + '\n')
        ret_info.append("pretrained embeddings url: " +
                        str(self.pretrained_embeddings_url) + '\n\n')

        return ''.join(ret_info)


class ZhFastTextWiki(EmbeddingParams):
    def __init__(self):
        super(ZhFastTextWiki, self).__init__()
        self.vec_dim = 300

        fast_text_vecs_wiki_zh_dir = os.path.join(self.pretrained_embeddings_root,
                                                  'fast_text_vectors_wiki.zh.vec')
        self.raw_pretrained_embeddings_url = os.path.join(fast_text_vecs_wiki_zh_dir,
                                                          'wiki.zh.vec')
        self.pretrained_embeddings_url = os.path.join(fast_text_vecs_wiki_zh_dir,
                                                      'processed.wiki.zh.vec')

    def __str__(self):
        ret_info = list()
        ret_info.append('================== '+self.current_classname+' ==================\n')
        super_str = super(ZhFastTextWiki, self).__str__()
        return ''.join(ret_info) + super_str


class TencentAIZhEmbeddings(EmbeddingParams):
    def __init__(self):
        super(TencentAIZhEmbeddings, self).__init__()
        self.vec_dim = 200
        self.start_tag = '</s>'
        self.end_tag = '</s>'

        tencent_ailab_zh_dir = os.path.join(self.pretrained_embeddings_root,
                                            'tencent_ailab_zh_embeddings')
        self.raw_pretrained_embeddings_url = os.path.join(tencent_ailab_zh_dir,
                                                          'tencent.ailab.zh.vecs')
        self.pretrained_embeddings_url = os.path.join(tencent_ailab_zh_dir,
                                                      'processed.tencent.ailab.zh.vecs')

    def __str__(self):
        ret_info = list()
        ret_info.append('================== '+self.current_classname+' ==================\n')
        super_str = super(TencentAIZhEmbeddings, self).__str__()
        return ''.join(ret_info) + super_str


embedding_name_abbr_full = {'zh-fasttext-wiki': ZhFastTextWiki().__class__.__name__,
                            'zh-tencent-ailab': TencentAIZhEmbeddings().__class__.__name__}
embedding_name_full_abbr = {v: k for k, v in embedding_name_abbr_full.items()}
available_embeddings = ['zh-fasttext-wiki', 'zh-tencent-ailab']


def get_embedding_params(embedding_name):
    if embedding_name == available_embeddings[0]:
        return ZhFastTextWiki()
    elif embedding_name == available_embeddings[1]:
        return TencentAIZhEmbeddings()
    else:
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() function, embedding_name value error.')


if __name__ == '__main__':
    print(EmbeddingParams())
    print(ZhFastTextWiki())
    print(TencentAIZhEmbeddings())
