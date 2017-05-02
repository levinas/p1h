from __future__ import print_function

import operator
import os
import warnings
import numpy as np

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.svm import *

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from xgboost import XGBRegressor
    from xgboost import XGBClassifier


def get_model(model_or_name, threads=-1, classification=False):
    regression_models = {
        'xgboost': (XGBRegressor(nthread=threads), 'XGBRegressor'),
        'randomforest': (RandomForestRegressor(n_estimators=100, n_jobs=threads), 'RandomForestRegressor'),
        'adaboost': (AdaBoostRegressor(), 'AdaBoostRegressor'),
        'linear': (LinearRegression(), 'LinearRegression'),
        'elasticnet': (ElasticNet(), 'ElasticNet'),
        'lasso': (Lasso(), 'Lasso'),
        'ridge': (Ridge(), 'Ridge')
    }

    classification_models = {
        'xgboost': (XGBClassifier(nthread=threads), 'XGBClassifier'),
        'randomforest': (RandomForestClassifier(n_estimators=100, n_jobs=threads), 'RandomForestClassifier'),
        'adaboost': (AdaBoostClassifier(), 'AdaBoostClassifier'),
        'logistic': (LogisticRegression(), 'LogisticRegression'),
        'gaussian': (GaussianProcessClassifier(), 'GaussianProcessClassifier'),
        'knn': (KNeighborsClassifier(), 'KNeighborsClassifier'),
        'bayes': (GaussianNB(), 'GaussianNB'),
        'svm': (SVC(), 'SVC')
    }

    if isinstance(model_or_name, str):
        if classification:
            model_and_name = classification_models.get(model_or_name.lower())
        else:
            model_and_name = regression_models.get(model_or_name.lower())
        if not model_and_name:
            raise Exception("unrecognized regression model: '{}'".format(model_or_name))
        else:
            model, name = model_and_name
    else:
        model = model_or_name
        name = re.search("\w+", str(model)).group(0)

    return model, name


def score_format(metric, score, eol=''):
    return '{:<25} = {:.5f}'.format(metric, score) + eol


def top_important_features(model, feature_names, n_top=100):
    if hasattr(model, "booster"): # XGB
        fscore = model.booster().get_fscore()
        fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
        features = [(v, feature_names[int(k[1:])]) for k,v in fscore]
        top = features[:n_top]
    else:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        else:
            if hasattr(model, "coef_"):
                fi = model.coef_
            else:
                return
        features = [(f, n) for f, n in zip(fi, feature_names)]
        top = sorted(features, key=lambda f:abs(f[0]), reverse=True)[:n_top]
    return top


def sprint_features(top_features, n_top=100):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= n_top:
            break
        str += '{}\t{:.5f}\n'.format(feature[1], feature[0])
    return str


def discretize(y, bins=4):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    y_discrete = np.digitize(y, thresholds)
    return y_discrete, thresholds


def regress(model, data, cv=5, threads=-1, prefix=''):
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model, name = get_model(model, threads)
    mat = data.as_matrix()
    x, y = mat[:, 1:], mat[:, 0]
    feature_labels = data.columns.tolist()[1:]

    train_scores, test_scores = [], []
    tests, preds = None, None

    y_discrete, thresholds = discretize(y, bins=4)
    print('Quartiles of y:', ['{:.2g}'.format(x) for x in thresholds], end='\n\n')

    print('>', name)
    print('Cross validation:')
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y_discrete)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        train_scores.append(model.score(x_train, y_train))
        test_scores.append(model.score(x_test, y_test))
        print("  fold {}/{}: score = {:.3f}".format(i+1, cv, model.score(x_test, y_test)))
        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test

    print('Average validation metrics:')
    scores_fname = "{}.{}.scores".format(prefix, name)
    metric_names = 'r2_score explained_variance_score mean_absolute_error mean_squared_error'.split()
    with open(scores_fname, "w") as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    top_features = top_important_features(model, feature_labels)
    if top_features is not None:
        fea_fname = "{}.{}.features".format(prefix, name)
        with open(fea_fname, "w") as fea_file:
            fea_file.write(sprint_features(top_features))


def classify(model, data, cv=5, threads=-1, prefix=''):
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model, name = get_model(model, threads)
    mat = data.as_matrix()
    x, y = mat[:, 1:], mat[:, 0]
    feature_labels = data.columns.tolist()[1:]

    train_scores, test_scores = [], []
    tests, preds = None, None
    probas = None

    print('>', name)
    print('Cross validation:')
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
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
        if hasattr(model, "predict_proba"):
            probas_ = model.fit(x_train, y_train).predict_proba(x_test)
            probas = np.concatenate((probas, probas_)) if probas is not None else probas_

    roc_auc_core = None
    if probas is not None:
        fpr, tpr, thresholds = metrics.roc_curve(tests, probas[:, 1], pos_label=0)
        roc_auc_score = metrics.auc(fpr, tpr)
        roc_fname = "{}.{}.ROC".format(prefix, name)
        with open(roc_fname, "w") as roc_file:
            roc_file.write('\t'.join(['Threshold', 'FPR', 'TPR'])+'\n')
            for ent in zip(thresholds, fpr, tpr):
                roc_file.write('\t'.join("{0:.5f}".format(x) for x in list(ent))+'\n')

    print('Average validation metrics:')
    scores_fname = "{}.{}.scores".format(prefix, name)
    metric_names = 'accuracy_score f1_score precision_score recall_score log_loss'.split()
    with open(scores_fname, "w") as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    top_features = top_important_features(model, feature_labels)
    if top_features is not None:
        fea_fname = "{}.{}.features".format(prefix, name)
        with open(fea_fname, "w") as fea_file:
            fea_file.write(sprint_features(top_features))
