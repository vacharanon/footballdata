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

X = X.loc[X['Year'] >= 2011]
Y = X['FTR']
del X['Lost']
del X['Draw']
del X['Won']
del X['FTR']
