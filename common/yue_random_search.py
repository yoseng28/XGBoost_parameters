"""
-------------------------------------------------
   Author :       yoseng
   Date：         2019/4/9
   Description :
-------------------------------------------------
"""
import time

from pandas import DataFrame
from sklearn.model_selection import RandomizedSearchCV


def random_search(clf, opt_params, x, y):
    print('******************************************************')
    print('开始随机搜索......')
    start = time.clock()
    clf = RandomizedSearchCV(clf, opt_params, cv=5, scoring='neg_log_loss', return_train_score=True)
    clf.fit(x, y)
    end = time.clock()
    print('随机搜索耗时：', end - start)
    print('best_param:', clf.best_params_)
    print('best_score:', clf.best_score_)
    print('best_estimator:', clf.best_estimator_)
    DataFrame(clf.cv_results_).to_csv('result/random_search.csv')
    print('结果:random_search.csv导出成功！')
    print('******************************************************')

