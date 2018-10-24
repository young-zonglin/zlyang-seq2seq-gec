from configs import available_models
from models import AttnSeq2SeqModel
from models import BasicModel


class ModelFactory:
    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == available_models[0]:
            return AttnSeq2SeqModel()
        else:
            return BasicModel()
