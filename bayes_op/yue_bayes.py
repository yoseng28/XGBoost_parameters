"""
-------------------------------------------------
   Author :       yoseng
   Date：         2019/4/9
   Description :
-------------------------------------------------
"""
import time

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from XGBoost_parameters.xgb.yue_xgb_params import bayes_params


def xgb_evaluate(max_depth, min_child_weight, subsample, colsample_bytree, n_estimators, reg_alpha, reg_lambda, train_X,
           train_y):
    xgb = XGBClassifier(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=2019
    )
    cv_result = cross_val_score(xgb, train_X, train_y, scoring='neg_log_loss', cv=5)
    return cv_result.mean()


def bayes_optimize(x, y, n_iter, init_points):
    def yue_cv(max_depth, min_child_weight, subsample, colsample_bytree, n_estimators, reg_alpha, reg_lambda):
        return xgb_evaluate(
            max_depth=int(max_depth),
            min_child_weight=int(min_child_weight),
            subsample=min(subsample, 0.999),
            colsample_bytree=min(colsample_bytree, 0.999),
            n_estimators=int(n_estimators),
            reg_alpha=min(reg_alpha, 0.999),
            reg_lambda=min(reg_lambda, 0.999),
            train_X=x,
            train_y=y,
        )

    optimizer = BayesianOptimization(
        f=yue_cv,
        pbounds=bayes_params,
        random_state=2019,
        verbose=2
    )
    print('******************************************************')
    print('开始贝叶斯优化......')
    start = time.clock()
    optimizer.maximize(n_iter, init_points)
    end = time.clock()
    print('贝叶斯优化耗时：', end - start)
    print("贝叶斯优化结果:", optimizer.max)
    print('******************************************************')
