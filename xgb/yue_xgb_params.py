"""
-------------------------------------------------
   Author :       yoseng
   Date：         2019/4/9
   Description :
-------------------------------------------------
"""
import numpy as np

# 随机搜索、网格搜索参数范围
common_params = {
    'max_depth': range(2, 20, 2),
    'min_child_weight': range(1, 10, 1),
    'subsample': np.arange(0.3, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0),
    'n_estimators': range(20, 200, 10),
    'reg_lambda': np.arange(0.1, 1.0, 0.2),
    'reg_alpha': np.arange(0.1, 1.0, 0.2),
}

# 贝叶斯优化参数范围
bayes_params = {
    'max_depth': (2, 20),
    'min_child_weight': (1, 10),
    'subsample': (0.3, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'n_estimators': (20, 200),
    'reg_alpha': (0.1, 1.0),
    'reg_lambda': (0.1, 1.0)
}
