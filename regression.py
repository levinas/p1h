from __future__ import print_function

import numpy as np

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import *
from sklearn.ensemble import *
from xgboost import XGBRegressor


def get_model(model_or_name, threads=-1):
    models = {
        'xgboost': (XGBRegressor(nthread=threads), 'XGBoost'),
        'randomforest': (RandomForestRegressor(n_estimators=100, n_jobs=threads), 'RandomForest'),
        'adaboost': (AdaBoostRegressor(), 'Adaboost'),
        'elasticnet': (ElasticNet(), 'ElasticNet'),
        'linear': (LinearRegression(), 'LinearRegression'),
        'lasso': (Lasso(), 'Lasso'),
        'ridge': (Ridge(), 'Ridge')
    }

    if isinstance(model_or_name, str):
        model_and_name = models.get(model_or_name.lower())
        if not model_and_name:
            raise Exception("unrecognized model: '{}'".format(model_or_name))
        else:
            model, name = model_and_name
    else:
        model = model_or_name
        name = re.search("\w+", str(model)).group(0)

    return model, name


def score_format(metric, score, eol=''):
    return '{:<25} = {:.5f}'.format(metric, score) + eol


def regress(model, x, y, cv=5, threads=-1):
    model, name = get_model(model, threads)

    train_scores, test_scores = [], []
    tests, preds = None, None

    print('>', name)
    print('Cross validation:')
    skf = KFold(n_splits=cv, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        train_scores.append(model.score(x_train, y_train))
        test_scores.append(model.score(x_test, y_test))
        print("  fold {}/{}: score = {:.3f}".format(i+1, cv, model.score(x_test, y_test)))

        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test

    metric_names = 'r2_score explained_variance_score mean_absolute_error mean_squared_error'.split()
    print('Average validation metrics:')
    for m in metric_names:
        try:
            s = getattr(metrics, m)(tests, preds)
            print(' ', score_format(m, s))
        except Exception:
            pass
    print()
