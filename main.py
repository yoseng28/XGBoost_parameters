"""
-------------------------------------------------
   Author :       yoseng
   Date：         2019/4/9
   Description :
-------------------------------------------------
"""
from sklearn import datasets

from XGBoost_parameters.xgb.yue_xgb_clf import xgb_clf
from XGBoost_parameters.xgb.yue_xgb_params import common_params
from XGBoost_parameters.bayes_op.yue_bayes import bayes_optimize
from XGBoost_parameters.common.yue_grid_search import gird_search
from XGBoost_parameters.common.yue_random_search import *

if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    train_X = data.data
    train_y = data.target
    # 随机搜索
    # random_search(xgb_clf(), common_params, train_X, train_y)
    # 网格搜索
    # gird_search(xgb_clf(), common_params, train_X, train_y)

    # 贝叶斯优化
    bayes_optimize(train_X, train_y, n_iter=30, init_points=2)
