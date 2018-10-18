import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from football_loader import football_loader, metrics

league = 'english'
validate_year = 2015
test_year = 2016
train_year = 2011
df = football_loader.load_league_csv(league)
# teams = df.loc[df['Year'] == validate_year, 'HomeTeam']
# teams = teams.unique()
teams = ['Arsenal']
teams.sort()
teams = football_loader.make_features(df, teams, train_year, validate_year, test_year)
df_league = None
classes = ['Draw', 'Lose', 'Win']
for team in teams:
    X = teams[team][0]
    Y = teams[team][1]
    # split data into train - test sets
    x_train = X[(X['Year'] < validate_year)]
    y_train = Y[(X['Year'] < validate_year)]
    x_validate = X[(X['Year'] >= validate_year) & (X['Year'] < test_year)]
    y_validate = Y[(X['Year'] >= validate_year) & (X['Year'] < test_year)]
    x_test = X[(X['Year'] >= test_year)]
    y_test = Y[(X['Year'] >= test_year)]
    if len(y_train) <= 0:
        print(f'skip {team}')
        continue
    # clf = RandomForestClassifier(n_estimators=10)
    lr = LogisticRegression()
    lr.fit(x_train, y_train['Result'])

    # in-sample test
    y_insample_pred = lr.predict(x_train)

    y_test_pred = lr.predict(x_test)

    # result = x_test
    # result.loc[x_test.index, 'Predict'] = y_pred

    # result['Actual'] = Y['Result']
    # result['Team'] = team
    # x = X[X['Predict'] != '']
    # print(f"{team} prediction accuracy are: ",  accuracy_score(y_train, y_insample_pred) * 100, "in-sample, ", accuracy_score(y_test, y_pred) * 100, " out-sample")
    print(f"{team} prediction accuracy are: ", accuracy_score(y_test, y_test_pred) * 100)
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    metrics.plot_confusion_matrix(cm, classes, title=team)
    # print(cm)
    # y_test_score = lr.predict_proba(x_test)
    y_test_score = lr.decision_function(x_test)
    #     print(y_test_pred)
    y_test_pred_binarized = label_binarize(y_test_pred, classes=classes)
    #     print(y_test_pred_binarized)
    fpr = {}
    tpr = {}
    roc_auc = {}
    # print(y_test_pred_binarized[:, i], y_test_score[:, i])
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_pred_binarized[:, i], y_test_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(classes[i], 'False positive rate', fpr[i])
        print(classes[i], 'True positive rate', tpr[i])
        print(classes[i], 'ROCAUC', roc_auc[i])
        # metrics.plot_roc_auc(fpr[i], tpr[i], roc_auc[i], classes[i] + ' ROC curve')
    # metrics.plot_roc_auc_multi(fpr, tpr, roc_auc, classes)

    # if df_league is None:
    #     df_league = x
    # else:
    #     df_league = df_league.append(x, ignore_index=True, sort=False)

# print("Overall accuracy is ", accuracy_score(df_league['Actual'], df_league['Predict'])*100)
