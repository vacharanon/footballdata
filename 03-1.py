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
    df_year = pd.read_csv(f'fulldata/{league}/' + file
                          # skiprows=1,
                          # index_col=None,
                          # names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']
                          )
    df_year.reset_index(drop=True, inplace=True)
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
# df = df[['Year', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
#          "B365H", "B365D", "B365A"]]

predict_year = 2017
train_year = 2011
teams = ['Ath Madrid']
# teams = np.unique(df.loc[df['Year'] == predict_year, 'HomeTeam'].values)
teams.sort()
for team in teams:

    df_team = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    X = pd.DataFrame(
        data={
            'Year': df_team['Year'],
            'Date': df_team['Date'],
            'Team': team,
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
    #     X['B365Max'] = np.maximum(np.maximum(df_team['B365H'], df_team['B365A']), df_team['B365D'])
    #     X['B365Say'] = np.where(X['HomeMatch'],
    #                             # home match
    #                             np.where(X['B365Max'] == df_team['B365H'], -1,
    #                                      np.where(X['B365Max'] == df_team['B365A'], 1,
    #                                               0)),
    #                             # away match
    #                             np.where(X['B365Max'] == df_team['B365H'], 1,
    #                                      np.where(X['B365Max'] == df_team['B365A'], -1,
    #                                               0))
    #                             )

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

        xx['Last3AgainstThisOpponentWon'] = xx['Won'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['Last3AgainstThisOpponentDraw'] = xx['Draw'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)

        xx['LastAgainstThisOpponentWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['LastAgainstThisOpponentDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)

        # restore index
        xx = xx.set_index('idx')

        # assign back to the big dataframe
        X.loc[xx.index, 'Last5AgainstThisOpponentWon'] = xx['Last5AgainstThisOpponentWon']
        X.loc[xx.index, 'Last5AgainstThisOpponentDraw'] = xx['Last5AgainstThisOpponentDraw']
        X.loc[xx.index, 'Last3AgainstThisOpponentWon'] = xx['Last3AgainstThisOpponentWon']
        X.loc[xx.index, 'Last3AgainstThisOpponentDraw'] = xx['Last3AgainstThisOpponentDraw']
        X.loc[xx.index, 'LastAgainstThisOpponentWon'] = xx['LastAgainstThisOpponentWon']
        X.loc[xx.index, 'LastAgainstThisOpponentDraw'] = xx['LastAgainstThisOpponentDraw']

    # find recent forms
    idx = X.index
    xx = X.set_index('Date')
    xx['idx'] = idx
    xx['Last5Won'] = xx['Won'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['Last5Draw'] = xx['Draw'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['Last3Won'] = xx['Won'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['Last3Draw'] = xx['Draw'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastMatch'] = np.where(xx['LastWon'] == 1, 1, np.where(xx['LastDraw'] == 1, 0, -1))
    # restore index
    xx = xx.set_index('idx')

    # assign back to the big dataframe
    X.loc[xx.index, 'Last5Won'] = xx['Last5Won']
    X.loc[xx.index, 'Last5Draw'] = xx['Last5Draw']
    X.loc[xx.index, 'Last3Won'] = xx['Last3Won']
    X.loc[xx.index, 'Last3Draw'] = xx['Last3Draw']
    X.loc[xx.index, 'LastWon'] = xx['LastWon']
    X.loc[xx.index, 'LastDraw'] = xx['LastDraw']
    X.loc[xx.index, 'LastMatch'] = xx['LastMatch']

    # replace nan with 0
    # TODO: better way to handle nan
    X.loc[np.isnan(X['Last5AgainstThisOpponentWon']), 'Last5AgainstThisOpponentWon'] = 0
    X.loc[np.isnan(X['Last5AgainstThisOpponentDraw']), 'Last5AgainstThisOpponentDraw'] = 0
    X.loc[np.isnan(X['Last3AgainstThisOpponentWon']), 'Last3AgainstThisOpponentWon'] = 0
    X.loc[np.isnan(X['Last3AgainstThisOpponentDraw']), 'Last3AgainstThisOpponentDraw'] = 0
    X.loc[np.isnan(X['LastAgainstThisOpponentWon']), 'LastAgainstThisOpponentWon'] = 0
    X.loc[np.isnan(X['LastAgainstThisOpponentDraw']), 'LastAgainstThisOpponentDraw'] = 0
    #     X.loc[np.isnan(X['Last5Won']), 'Last5Won'] = 0
    #     X.loc[np.isnan(X['Last5Draw']), 'Last5Draw'] = 0
    #     X.loc[np.isnan(X['Last3Won']), 'Last3Won'] = 0
    #     X.loc[np.isnan(X['Last3Draw']), 'Last3Draw'] = 0
    #     X.loc[np.isnan(X['LastWon']), 'LastWon'] = 0
    #     X.loc[np.isnan(X['LastDraw']), 'LastDraw'] = 0

    # restrict training data (too old data may not be irrelevance)
    X = X.loc[X['Year'] >= train_year]
    Y = X[['Opponent', 'Result']]

    # remove duplicate features
    del X['LastWon']
    del X['LastDraw']

    # prevent future leaks
    del X['Result']
    del X['Lost']
    del X['Draw']
    del X['Won']
    del X['FTR']
    del X['Date']
    #     del X['B365Max']

    # split data into train - test sets
    x_train = X[(X['Year'] < predict_year)]
    y_train = Y[(X['Year'] < predict_year)]
    x_test = X[(X['Year'] >= predict_year)]
    y_test = Y[(X['Year'] >= predict_year)]

    # split prediction by opponent
    # construct decision tree
    x_train_opponents = x_train.groupby('Opponent')
    y_train_opponents = y_train.groupby('Opponent')
    #     X_test_opponents = x_test.groupby('Opponent')
    #     Y_test_opponents = y_test.groupby('Opponent')
    #     x_test_teams = X_test_opponents.groups.keys()
    x_test_teams = ['Villarreal']
    X['Predict'] = ''
    #     os.makedirs(f'decision_tree/{league}/{predict_year}/{team}/', exist_ok=True)
    #     for key, x_train_opponent in x_train_opponents:
    for key in x_test_teams:
        #         if key not in x_test_teams:
        #             continue
        x_train_opponent = x_train_opponents.get_group(key)
        #         X_test_opponent = X_test_opponents.get_group(key)
        y_train_opponent = y_train_opponents.get_group(key)
        #         Y_test_opponent = Y_test_opponents.get_group(key)

        del y_train_opponent['Opponent']
        #         del Y_test_opponent['Opponent']
        del x_train_opponent['Opponent']
        #         del X_test_opponent['Opponent']
        del x_train_opponent['Year']
        #         del X_test_opponent['Year']
        del x_train_opponent['Team']
        #         del X_test_opponent['Team']

        clf = DecisionTreeClassifier(
            criterion="entropy",
            random_state=100,
            min_samples_leaf=3
        )
        clf.fit(x_train_opponent, y_train_opponent)

        # in-sample test

#         Y_pred = clf.predict(X_test_opponent)
