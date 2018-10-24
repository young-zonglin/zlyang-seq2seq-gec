import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULT_SAVE_DIR = os.path.join(PROJECT_ROOT, 'result')

TRAIN_RECORD_FNAME = 'a.train.info.record'
# 注意：后缀应为png，jpeg之类
MODEL_VIS_FNAME = 'a.model.visual.png'

MATCH_SINGLE_QUOTE_STR = r'[^a-zA-Z]*(\')[a-zA-Z ]*(\')[^a-zA-Z]+'

SEPARATOR = '\t'

GENERAL_OPEN_ENCODING = 'utf-8'
GENERAL_SAVE_ENCODING = 'utf-8'
