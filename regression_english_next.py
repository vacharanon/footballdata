import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, log_loss
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier
import pandas as pd
import operator
from football_loader.football_loader import load_league_csv, make_features, load_next_csv
from football_loader import metrics

league = 'english'
# validate_year = 2016
# test_year = 2017
train_year = 2011
# validate_year = 2017
test_year = 2018
df = load_league_csv(league)
div = df['Div'].values[0]
# teams = df.loc[(df['Year'] == validate_year) | (df['Year'] == test_year), 'HomeTeam']
teams = df.loc[(df['Year'] == test_year), 'HomeTeam']
teams = teams.unique()
teams.sort()
# teams = ['Chelsea']
# df = df[(df['HomeTeam'] == 'Chelsea') | (df['AwayTeam'] == 'Chelsea')]
# df.reset_index(inplace=True)

# insert next matches
df_next = load_next_csv(league, div)
df = df.append(df_next, sort=False)
df.reset_index(inplace=True)

teams = make_features(df, teams)
classes = ['Draw', 'Lose', 'Win']
team_result = {}
total = None
for team in teams:
    # for team in ['Arsenal']:
    X = teams[team][0]
    Y = teams[team][1]
    # split data into train - validate - test sets
    # x_train = X[(X['Year'] < validate_year)]
    # y_train = Y[(X['Year'] < validate_year)]
    x_train = X[(X['Year'] >= train_year) & (Y['Result'] != '')]
    y_train = Y[(X['Year'] >= train_year) & (Y['Result'] != '')]
    # x_validate = X[(X['Year'] >= validate_year) & (X['Year'] < test_year)]
    # y_validate = Y[(X['Year'] >= validate_year) & (X['Year'] < test_year)]
    # x_test = X[(X['Year'] >= test_year) & (X['Result'] != '')]
    # y_test = Y[(X['Year'] >= test_year) & (X['Result'] != '')]
    x_next = X[(X['Year'] >= test_year) & (Y['Result'] == '')]
    # y_next = Y[(X['Year'] >= test_year) & (X['Result'] != '')]
    # if len(x_train) <= 0 or len(x_test) <= 0 or len(x_validate) <= 0:
    #     print(f'skip {team}')
    #     continue
    if len(df.loc[x_next.index]) <= 0: continue
    homeAwayTeams = df.loc[x_next.index][['HomeTeam', 'AwayTeam']].values[0]
    # validate_accuracies = {}
    # test_accuracies = {}

    lr = LogisticRegression()
    lr.fit(x_train, y_train['Result'])
    # y_validate_pred = lr.predict(x_validate)
    # y_validate_pred = lr.predict(y_validate)
    # validate_accuracies['LogisticRegression'] = accuracy_score(y_validate, y_validate_pred) * 100
    # y_test_pred = lr.predict(x_test)
    y_next_ped = lr.predict(x_next)
    # print(y_next_ped)
    prob = np.round(lr.predict_proba(x_next)*100, 2)[0]
    print(homeAwayTeams[0], '(H) vs', homeAwayTeams[1], '(A) ->', team, y_next_ped[0], list(zip(classes, prob)))
   
    team_result[homeAwayTeams[0]] = y_next_ped[0]
    # test_accuracies['LogisticRegression'] = accuracy_score(y_test, y_test_pred) * 100
    # d = pd.DataFrame(
    #     data={
    #         'pred': y_test_pred,
    #         'actual': y_test['Result'].values
    #     }
    # )
    # if total is None:
    #     total = d
    # else:
    #     total = total.append(d)
    # for (k, v) in validate_accuracies.items():
    #     print(f"{k}: {team} validation accuracy are: {v}")
    #     print(f"{k}: {team} test accuracy are: {test_accuracies[k]}")
    # Let's try next match

print(team_result)