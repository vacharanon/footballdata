import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from football_loader import football_loader

league = 'italian'
df = football_loader.load_league_csv(league)
validate_year = 2015
test_year = 2016
train_year = 2011
# teams = ['Milan', 'Juventus']
teams = df.loc[df['Year'] == validate_year, 'HomeTeam']
teams = teams.unique()
# print(df.info())
# teams = np.unique(teams)
# teams = np.unique(df.loc[df['Year'] == predict_year, 'HomeTeam'].values)
# teams.sort()
teams = football_loader.make_features(df, teams, train_year, validate_year)
df_league = None

for team in teams:
    X = teams[team][0]
    Y = teams[team][1]
    # split data into train - test sets
    x_train = X[(X['Year'] < validate_year)]
    y_train = Y[(X['Year'] < validate_year)]
    x_test = X[(X['Year'] >= validate_year)]
    y_test = Y[(X['Year'] >= validate_year)]
    if len(y_train) <= 0:
        print(f'skip {team}')
        continue
    # clf = RandomForestClassifier(n_estimators=10)
    clf = LogisticRegression()
    clf.fit(x_train, y_train['Result'])

    # in-sample test
    y_insample_pred = clf.predict(x_train)

    y_pred = clf.predict(x_test)

    # result = x_test
    # result.loc[x_test.index, 'Predict'] = y_pred

    # result['Actual'] = Y['Result']
    # result['Team'] = team
    # x = X[X['Predict'] != '']
    print(f"{team} prediction accuracy are: ",  accuracy_score(y_train, y_insample_pred) * 100, "in-sample, ", accuracy_score(y_test, y_pred) * 100, " out-sample")

    # if df_league is None:
    #     df_league = x
    # else:
    #     df_league = df_league.append(x, ignore_index=True, sort=False)

# print("Overall accuracy is ", accuracy_score(df_league['Actual'], df_league['Predict'])*100)
