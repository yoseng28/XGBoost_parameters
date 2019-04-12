"""
-------------------------------------------------
   Author :       yoseng
   Date：         2019/4/9
   Description :
-------------------------------------------------
"""
from xgboost import XGBClassifier


def xgb_clf():
    # XGB固定参数
    clf_params = {
        'random_state': 2019,
        'learning_rate': 0.1,
        'gamma': 0.1,
    }
    xgb = XGBClassifier(**clf_params)
    return xgb
