import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier
import numpy as np
import pandas as pd
import operator
from football_loader import football_loader, metrics

league = 'italian'
validate_year = 2016
test_year = 2017
train_year = 2005
# train_year = 2009
df = football_loader.load_league_csv(league)
# teams = df.loc[(df['Year'] == validate_year) | (df['Year'] == test_year), 'HomeTeam']
# teams = teams.unique()
# teams.sort()
teams = ['Bologna']
teams = football_loader.make_features(df, teams)
df_league = None
classes = ['Draw', 'Lose', 'Win']

total = None
for team in teams:
# for team in ['Bologna']:
    X = teams[team][0]
    Y = teams[team][1]
    # split data into train - validate - test sets
    x_train = X[(X['Year'] >= train_year) & (X['Year'] < validate_year)]
    y_train = Y[(X['Year'] >= train_year) & (X['Year'] < validate_year)]
    x_validate = X[(X['Year'] >= validate_year) & (X['Year'] < test_year)]
    y_validate = Y[(X['Year'] >= validate_year) & (X['Year'] < test_year)]
    x_test = X[(X['Year'] >= test_year)]
    y_test = Y[(X['Year'] >= test_year)]
    if len(x_train) <= 0 or len(x_test) <= 0 or len(x_validate) <= 0:
        print(f'skip {team}')
        continue

    validate_accuracies = {}
    test_accuracies = {}

    lr = LogisticRegression()
    lr.fit(x_train, y_train['Result'])
    y_validate_pred = lr.predict(x_validate)
    validate_accuracies['LogisticRegression'] = accuracy_score(y_validate, y_validate_pred) * 100
    y_test_pred = lr.predict(x_test)
    test_accuracies['LogisticRegression'] = accuracy_score(y_test, y_test_pred) * 100
    d = pd.DataFrame(
        data={
            'pred': y_test_pred,
            'actual': y_test['Result'].values
        },
        index=pd.MultiIndex.from_tuples(
            list(zip(np.repeat('LogisticRegression', len(y_test.index)), y_test.index.values)),
            names=['predictor', 'match_id'])
    )
    if total is None:
        total = d
    else:
        total = total.append(d)

    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(x_train, y_train['Result'])
    y_validate_pred = rfc.predict(x_validate)
    validate_accuracies['RandomForestClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    y_test_pred = rfc.predict(x_test)
    test_accuracies['RandomForestClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    d = pd.DataFrame(
        data={
            'pred': y_test_pred,
            'actual': y_test['Result'].values
        },
        index=pd.MultiIndex.from_tuples(
            list(zip(np.repeat('RandomForestClassifier', len(y_test.index)), y_test.index.values)),
            names=['predictor', 'match_id'])
    )
    if total is None:
        total = d
    else:
        total = total.append(d)

    abc = AdaBoostClassifier()
    abc.fit(x_train, y_train['Result'])
    y_validate_pred = abc.predict(x_validate)
    validate_accuracies['AdaBoostClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    y_test_pred = abc.predict(x_test)
    test_accuracies['AdaBoostClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    d = pd.DataFrame(
        data={
            'pred': y_test_pred,
            'actual': y_test['Result'].values
        },
        index=pd.MultiIndex.from_tuples(
            list(zip(np.repeat('AdaBoostClassifier', len(y_test.index)), y_test.index.values)),
            names=['predictor', 'match_id'])
    )
    if total is None:
        total = d
    else:
        total = total.append(d)

    bc = BaggingClassifier()
    bc.fit(x_train, y_train['Result'])
    y_validate_pred = bc.predict(x_validate)
    validate_accuracies['BaggingClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    y_test_pred = bc.predict(x_test)
    test_accuracies['BaggingClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    d = pd.DataFrame(
        data={
            'pred': y_test_pred,
            'actual': y_test['Result'].values
        },
        index=pd.MultiIndex.from_tuples(
            list(zip(np.repeat('BaggingClassifier', len(y_test.index)), y_test.index.values)),
            names=['predictor', 'match_id'])
    )
    if total is None:
        total = d
    else:
        total = total.append(d)

    etc = ExtraTreesClassifier()
    etc.fit(x_train, y_train['Result'])
    y_validate_pred = etc.predict(x_validate)
    validate_accuracies['ExtraTreesClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    y_test_pred = etc.predict(x_test)
    test_accuracies['ExtraTreesClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    d = pd.DataFrame(
        data={
            'pred': y_test_pred,
            'actual': y_test['Result'].values
        },
        index=pd.MultiIndex.from_tuples(
            list(zip(np.repeat('ExtraTreesClassifier', len(y_test.index)), y_test.index.values)),
            names=['predictor', 'match_id'])
    )
    if total is None:
        total = d
    else:
        total = total.append(d)

    gbc = GradientBoostingClassifier()
    gbc.fit(x_train, y_train['Result'])
    y_validate_pred = gbc.predict(x_validate)
    validate_accuracies['GradientBoostingClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    y_test_pred = gbc.predict(x_test)
    test_accuracies['GradientBoostingClassifier'] = accuracy_score(y_validate, y_validate_pred) * 100
    d = pd.DataFrame(
        data={
            'pred': y_test_pred,
            'actual': y_test['Result'].values
        },
        index=pd.MultiIndex.from_tuples(
            list(zip(np.repeat('GradientBoostingClassifier', len(y_test.index)), y_test.index.values)),
            names=['predictor', 'match_id'])
    )
    if total is None:
        total = d
    else:
        total = total.append(d)

    for (k, v) in validate_accuracies.items():
        print(f"{k}: {team} validation accuracy are: {v}")
        print(f"{k}: {team} test accuracy are: {test_accuracies[k]}")

    best = max(validate_accuracies.items(), key=operator.itemgetter(1))
    print('***Best validate classifier is', best[0], best[1])
    best = max(test_accuracies.items(), key=operator.itemgetter(1))
    print('***Best test classifier is', best[0], best[1])
    print('-------------------------')
    #     cm = confusion_matrix(y_validate, y_validate_pred, labels=classes)
    #     metrics.plot_confusion_matrix(cm, classes, title=team)

    if total is None:
        total = d
    else:
        total = total.append(d)
for key, groupByPredictor in total.groupby('predictor'):
    print(f"{key} overall accuracy is ", accuracy_score(groupByPredictor['actual'], groupByPredictor['pred'])*100)