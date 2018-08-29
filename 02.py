import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


df = None
files = os.listdir('fulldata/spanish')
files.sort()
for file in files:
    year = int(file.strip('.csv'))
    df_year = pd.read_csv('fulldata/spanish/' + file)
    df_year['Year'] = year
    df_year['Match'] = df_year.index + 1

    if df is None:
        df = df_year
    else:
        df = df.append(df_year, ignore_index=True, sort=False)

# print(len(df))
# print(df.shape)

# remove unused columns
df.reset_index(inplace=True)
df = df[['Year', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
ath_madrid = df[(df['HomeTeam'] == 'Ath Madrid') | (df['AwayTeam'] == 'Ath Madrid')]
X = pd.DataFrame(
    data={
        'Year': ath_madrid['Year'],
        'Date': ath_madrid['Date'],
        'HomeMatch': ath_madrid['HomeTeam'] == 'Ath Madrid'
    }
)
X['Opponent'] = np.where(X['HomeMatch'], ath_madrid['AwayTeam'], ath_madrid['HomeTeam'])
X['HalfTimeGoals'] = np.where(X['HomeMatch'], ath_madrid['HTHG'], ath_madrid['HTAG'])
X['HalfTimeOpponentGoals'] = np.where(X['HomeMatch'], ath_madrid['HTAG'], ath_madrid['HTHG'])
X['HalfTimeLead'] = X['HalfTimeGoals'] > X['HalfTimeOpponentGoals']
X['HalfTimeLeadMoreThanTwo'] = (X['HalfTimeGoals'] - X['HalfTimeOpponentGoals']) > 2
X['FullTimeGoals'] = np.where(X['HomeMatch'], ath_madrid['FTHG'], ath_madrid['FTAG'])
X['FullTimeOpponentGoals'] = np.where(X['HomeMatch'], ath_madrid['FTAG'], ath_madrid['FTHG'])
X['FTR'] = ath_madrid['FTR']
X['Won'] = np.where(X['HomeMatch'], ath_madrid['FTR'] == 'H', ath_madrid['FTR'] == 'A')
X['Draw'] = ath_madrid['FTR'] == 'D'
X['Lost'] = np.where(X['HomeMatch'], ath_madrid['FTR'] == 'A', ath_madrid['FTR'] == 'H')

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
    xx['Last5AgainstThisOpponentLost'] = xx['Lost'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
    # xx['Last5AgainstThisOpponentWon'] = xx['Won'].rolling(5).apply(lambda x: np.count_nonzero(x), raw=False)
    # xx['Last5AgainstThisOpponentDraw'] = xx['Draw'].rolling(5).apply(lambda x: np.count_nonzero(x), raw=False)

    xx['LastThisOpponentWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastThisOpponentDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
    xx['LastThisOpponentLost'] = xx['Lost'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)

    # restore index
    xx = xx.set_index('idx')

    # assign back to the big dataframe
    X.loc[xx.index, 'Last5AgainstThisOpponentWon'] = xx['Last5AgainstThisOpponentWon']
    X.loc[xx.index, 'Last5AgainstThisOpponentDraw'] = xx['Last5AgainstThisOpponentDraw']
    X.loc[xx.index, 'Last5AgainstThisOpponentLost'] = xx['Last5AgainstThisOpponentLost']
    X.loc[xx.index, 'LastThisOpponentWon'] = xx['LastThisOpponentWon']
    X.loc[xx.index, 'LastThisOpponentDraw'] = xx['LastThisOpponentDraw']
    X.loc[xx.index, 'LastThisOpponentLost'] = xx['LastThisOpponentLost']

# find recent forms
idx = X.index
xx = X.set_index('Date')
xx['idx'] = idx
xx['Last5Won'] = xx['Won'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
xx['Last5Draw'] = xx['Draw'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
xx['Last5Lost'] = xx['Lost'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
# restore index
xx = xx.set_index('idx')
# assign back to the big dataframe
X.loc[xx.index, 'Last5Won'] = xx['Last5Won']
X.loc[xx.index, 'Last5Draw'] = xx['Last5Draw']
X.loc[xx.index, 'Last5Lost'] = xx['Last5Lost']

X.loc[np.isnan(X['Last5AgainstThisOpponentWon']), 'Last5AgainstThisOpponentWon'] = 0
X.loc[np.isnan(X['Last5AgainstThisOpponentDraw']), 'Last5AgainstThisOpponentDraw'] = 0
X.loc[np.isnan(X['Last5AgainstThisOpponentLost']), 'Last5AgainstThisOpponentLost'] = 0
X.loc[np.isnan(X['LastThisOpponentWon']), 'LastThisOpponentWon'] = 0
X.loc[np.isnan(X['LastThisOpponentDraw']), 'LastThisOpponentDraw'] = 0
X.loc[np.isnan(X['LastThisOpponentLost']), 'LastThisOpponentLost'] = 0
X.loc[np.isnan(X['Last5Won']), 'Last5Won'] = 0
X.loc[np.isnan(X['Last5Draw']), 'Last5Draw'] = 0
X.loc[np.isnan(X['Last5Lost']), 'Last5Lost'] = 0

# X = X.loc[X['Year'] >= 2011]
Y = X[['Opponent', 'FTR']]
del X['Lost']
del X['Draw']
del X['Won']
del X['FTR']
del X['Date']

X_train = X[(X['Year'] >= 2011) & (X['Year'] <= 2016)]
Y_train = Y[(X['Year'] >= 2011) & (X['Year'] <= 2016)]
X_test = X[(X['Year'] >= 2017)]
Y_test = Y[(X['Year'] >= 2017)]

# construct decision tree
X_train_opponents = X_train.groupby('Opponent')
Y_train_opponents = Y_train.groupby('Opponent')
X_test_opponents = X_test.groupby('Opponent')
Y_test_opponents = Y_test.groupby('Opponent')
x_test_teams = X_test_opponents.groups.keys()
X['Predict'] = ''
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

    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
     max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train_opponent, Y_train_opponent)

    Y_pred = clf_entropy.predict(X_test_opponent)
    X.loc[X_test_opponent.index, 'Predict'] = Y_pred
    # print(Y_test_opponent)
X['Actual'] = Y['FTR']
x = X[X['Predict'] != '']
x.to_csv('./x.csv')
print('done')
