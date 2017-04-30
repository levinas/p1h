
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold


def score_format(metric, score, eol='\n'):
    return '{:<15} = {:.5f}'.format(metric, score) + eol


def regress(model, x, y, cv=5):
    train_scores, test_scores = [], []
    tests, preds = None, None
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        x_train, x_test = x[train_index], y[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        train_scores.append(model.score(x_train, y_train))
        test_scores.append(model.score(x_test, y_test))
        print("  fold #{}: score={:.3f}".format(i, clf.score(x_test, y_test)))

        y_pred = clf.predict(X_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test

        metric_names = 'mean_squared_error, mean_absolute_error, explained_variance_score, r2_score'.split()
        for m in metrics:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(score_format(m, s))
            except Exception:
                pass
