import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from subprocess import call

league = 'spanish'

df = None
files = os.listdir(f'fulldata/{league}')
files.sort()
for file in files:
    year = int(file.strip('.csv'))
    df_year = pd.read_csv(f'fulldata/{league}/' + file)
    df_year['Year'] = year
    df_year['Match'] = df_year.index + 1

    if df is None:
        df = df_year
    else:
        df = df.append(df_year, ignore_index=True, sort=False)

# print(len(df))
# print(df.shape)

# remove unused columns
df_league = None
df.reset_index(inplace=True)
df = df[['Year', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]

predict_year = 2017
# team = 'Ath Madrid'
teams = np.unique(df.loc[df['Year'] == predict_year, 'HomeTeam'].values)
teams.sort()
for team in teams:

    df_team = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    X = pd.DataFrame(
        data={
            'Year': df_team['Year'],
            'Date': df_team['Date'],
            'HomeMatch': df_team['HomeTeam'] == team
        }
    )
    X['Opponent'] = np.where(X['HomeMatch'], df_team['AwayTeam'], df_team['HomeTeam'])
    # X['HalfTimeGoals'] = np.where(X['HomeMatch'], df_team['HTHG'], df_team['HTAG'])
    # X['HalfTimeOpponentGoals'] = np.where(X['HomeMatch'], df_team['HTAG'], df_team['HTHG'])
    # X['HalfTimeLead'] = X['HalfTimeGoals'] > X['HalfTimeOpponentGoals']
    # X['HalfTimeLeadMoreThanTwo'] = (X['HalfTimeGoals'] - X['HalfTimeOpponentGoals']) > 2
    # X['FullTimeGoals'] = np.where(X['HomeMatch'], ath_madrid['FTHG'], ath_madrid['FTAG'])
    # X['FullTimeOpponentGoals'] = np.where(X['HomeMatch'], ath_madrid['FTAG'], ath_madrid['FTHG'])
    X['FTR'] = df_team['FTR']
    X['Won'] = np.where(X['HomeMatch'], df_team['FTR'] == 'H', df_team['FTR'] == 'A')
    X['Draw'] = df_team['FTR'] == 'D'
    X['Lost'] = np.where(X['HomeMatch'], df_team['FTR'] == 'A', df_team['FTR'] == 'H')
    X['Result'] = np.where(X['Won'], 'Win', (np.where(X['Lost'], 'Lose', 'Draw')))
    # X['SumGoals'] = X.groupby('Opponent')['FullTimeGoals'].transform(sum)

    # find number of times won against this opponent in last 5 meetings
    for key, groupByOpponent in X.groupby('Opponent'):
        # keep index as new a column, will be restored and assigned back to X later
        idx = groupByOpponent.index

        # make match day an index because rolling need an index date
        xx = groupByOpponent.set_index('Date')
        xx['idx'] = idx
        # shift to exclude self
        xx['Last5AgainstThisOpponentWon'] = xx['Won'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['Last5AgainstThisOpponentDraw'] = xx['Draw'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
        # xx['Last5AgainstThisOpponentLost'] = xx['Lost'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)

        xx['LastThisOpponentWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['LastThisOpponentDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
        # xx['LastThisOpponentLost'] = xx['Lost'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)

        # restore index
        xx = xx.set_index('idx')

        # assign back to the big dataframe
        X.loc[xx.index, 'Last5AgainstThisOpponentWon'] = xx['Last5AgainstThisOpponentWon']
        X.loc[xx.index, 'Last5AgainstThisOpponentDraw'] = xx['Last5AgainstThisOpponentDraw']
        # X.loc[xx.index, 'Last5AgainstThisOpponentLost'] = xx['Last5AgainstThisOpponentLost']
        X.loc[xx.index, 'LastThisOpponentWon'] = xx['LastThisOpponentWon']
        X.loc[xx.index, 'LastThisOpponentDraw'] = xx['LastThisOpponentDraw']
        # X.loc[xx.index, 'LastThisOpponentLost'] = xx['LastThisOpponentLost']

    # find recent forms
    idx = X.index
    xx = X.set_index('Date')
    xx['idx'] = idx
    xx['Last5Won'] = xx['Won'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['Last5Draw'] = xx['Draw'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
    # xx['Last5Lost'] = xx['Lost'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
    # restore index
    xx = xx.set_index('idx')
    # assign back to the big dataframe
    X.loc[xx.index, 'Last5Won'] = xx['Last5Won']
    X.loc[xx.index, 'Last5Draw'] = xx['Last5Draw']
    X.loc[xx.index, 'LastWon'] = xx['LastWon']
    X.loc[xx.index, 'LastDraw'] = xx['LastDraw']
    # X.loc[xx.index, 'Last5Lost'] = xx['Last5Lost']

    X.loc[np.isnan(X['Last5AgainstThisOpponentWon']), 'Last5AgainstThisOpponentWon'] = 0
    X.loc[np.isnan(X['Last5AgainstThisOpponentDraw']), 'Last5AgainstThisOpponentDraw'] = 0
    # X.loc[np.isnan(X['Last5AgainstThisOpponentLost']), 'Last5AgainstThisOpponentLost'] = 0
    X.loc[np.isnan(X['LastThisOpponentWon']), 'LastThisOpponentWon'] = 0
    X.loc[np.isnan(X['LastThisOpponentDraw']), 'LastThisOpponentDraw'] = 0
    # X.loc[np.isnan(X['LastThisOpponentLost']), 'LastThisOpponentLost'] = 0
    X.loc[np.isnan(X['Last5Won']), 'Last5Won'] = 0
    X.loc[np.isnan(X['Last5Draw']), 'Last5Draw'] = 0
    # X.loc[np.isnan(X['Last5Lost']), 'Last5Lost'] = 0
    X.loc[np.isnan(X['LastWon']), 'LastWon'] = 0
    X.loc[np.isnan(X['LastDraw']), 'LastDraw'] = 0

    # X = X.loc[X['Year'] >= 2011]
    Y = X[['Opponent', 'Result']]
    del X['Result']
    del X['Lost']
    del X['Draw']
    del X['Won']
    del X['FTR']
    del X['Date']

    # x_train = X[(X['Year'] >= 2011) & (X['Year'] <= 2016)]
    # y_train = Y[(X['Year'] >= 2011) & (X['Year'] <= 2016)]
    X_train = X[(X['Year'] < predict_year)]
    Y_train = Y[(X['Year'] < predict_year)]
    X_test = X[(X['Year'] >= predict_year)]
    Y_test = Y[(X['Year'] >= predict_year)]

    # construct decision tree
    X_train_opponents = X_train.groupby('Opponent')
    Y_train_opponents = Y_train.groupby('Opponent')
    X_test_opponents = X_test.groupby('Opponent')
    Y_test_opponents = Y_test.groupby('Opponent')
    x_test_teams = X_test_opponents.groups.keys()
    X['Predict'] = ''
    os.makedirs(f'decision_tree/{league}/{predict_year}/{team}/', exist_ok=True)
    for key, X_train_opponent in X_train_opponents:
        if key not in x_test_teams:
            continue
        X_test_opponent = X_test_opponents.get_group(key)
        Y_train_opponent = Y_train_opponents.get_group(key)
        Y_test_opponent = Y_test_opponents.get_group(key)

        del Y_train_opponent['Opponent']
        del Y_test_opponent['Opponent']
        del X_train_opponent['Opponent']
        del X_test_opponent['Opponent']
        del X_train_opponent['Year']
        del X_test_opponent['Year']

        clf_entropy = DecisionTreeClassifier(
            criterion="gini",
            # criterion="entropy",
            random_state=100,
            # max_depth=3,
            min_samples_leaf=5
            # min_samples_leaf=3
        )
        clf_entropy.fit(X_train_opponent, Y_train_opponent)

        Y_pred = clf_entropy.predict(X_test_opponent)
        X.loc[X_test_opponent.index, 'Predict'] = Y_pred
        tree.export_graphviz(clf_entropy, out_file=f'decision_tree/{league}/{predict_year}/{team}/{key}.dot',
                             feature_names=X_test_opponent.columns.values,
                             class_names=clf_entropy.classes_)
        call(['dot', '-Tpng', f'decision_tree/{league}/{predict_year}/{team}/{key}.dot', '-o', f'decision_tree/{league}/{predict_year}/{team}/{key}.png'])

    X['Actual'] = Y['Result']
    x = X[X['Predict'] != '']
    x.to_csv(f'./decision_tree/{league}/{predict_year}/{team}/x.csv')
    print(f"{team} prediction accuracy is ", accuracy_score(x['Actual'], x['Predict'])*100)

    if df_league is None:
        df_league = x
    else:
        df_league = df_league.append(x, ignore_index=True, sort=False)
print(df_league.count())
print("Overall accuracy is ", accuracy_score(df_league['Actual'], df_league['Predict'])*100)
