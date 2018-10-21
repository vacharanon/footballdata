import os
import numpy as np
import pandas as pd
from datetime import datetime


def load_next_csv(league, div):
    df_next = pd.read_csv(f'next/{league}.csv',
                              engine='python')
    df_next = df_next[df_next['Div'] == div]
    df_next.reset_index(drop=True, inplace=True)
    df_next['Year'] = datetime.strptime(df_next['Date'].values[0], '%d/%m/%y').date().year
    df_next['Match'] = df_next.index + 1
    df_next['FTR'] = ''
    return df_next


def load_league_csv(league, start_year=2005):
    df = None
    files = os.listdir(f'fulldata/{league}')
    files.sort()
    for file in files:
        year = int(file.strip('.csv'))
        if year < start_year:
            continue
        df_year = pd.read_csv(f'fulldata/{league}/' + file,
                              engine='python',
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
    # df_league = None
    df.reset_index(inplace=True)
    # df = df[['Year', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
    #          "B365H", "B365D", "B365A"]]
    return df


def make_features(df, teams):
    # df_league = None
    ret = {}

    # if teams is None:
    #     teams = np.unique(df.loc[df['Year'] == predict_year, 'HomeTeam'].values)
    #     teams.sort()
    for team in teams:

        df_team = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        all = pd.DataFrame(
            data={
                'Year': df_team['Year'],
                'Date': df_team['Date'],
                'Team': team,
                'HomeMatch': df_team['HomeTeam'] == team
            }
        )
        all['Opponent'] = np.where(all['HomeMatch'], df_team['AwayTeam'], df_team['HomeTeam'])
        # X['HalfTimeGoals'] = np.where(X['HomeMatch'], df_team['HTHG'], df_team['HTAG'])
        # X['HalfTimeOpponentGoals'] = np.where(X['HomeMatch'], df_team['HTAG'], df_team['HTHG'])
        # X['HalfTimeLead'] = X['HalfTimeGoals'] > X['HalfTimeOpponentGoals']
        # X['HalfTimeLeadMoreThanTwo'] = (X['HalfTimeGoals'] - X['HalfTimeOpponentGoals']) > 2
        # X['FullTimeGoals'] = np.where(X['HomeMatch'], ath_madrid['FTHG'], ath_madrid['FTAG'])
        # X['FullTimeOpponentGoals'] = np.where(X['HomeMatch'], ath_madrid['FTAG'], ath_madrid['FTHG'])
        all['FTR'] = df_team['FTR']
        # all['Won'] = np.where(all['HomeMatch'], df_team['FTR'] == 'H', df_team['FTR'] == 'A')
        all['Won'] = np.where(df_team['FTR'] == '', False, np.where(all['HomeMatch'], df_team['FTR'] == 'H', df_team['FTR'] == 'A'))
        all['Draw'] = np.where(df_team['FTR'] == '', False, df_team['FTR'] == 'D')
        all['Lost'] = np.where(df_team['FTR'] == '', False, np.where(all['HomeMatch'], df_team['FTR'] == 'A', df_team['FTR'] == 'H'))
        all['Result'] = np.where(df_team['FTR'] == '', '', np.where(all['Won'], 'Win', (np.where(all['Lost'], 'Lose', 'Draw'))))
        # X['SumGoals'] = X.groupby('Opponent')['FullTimeGoals'].transform(sum)
        all['B365Max'] = np.maximum(np.maximum(df_team['B365H'], df_team['B365A']), df_team['B365D'])
        all['B365Min'] = np.minimum(np.minimum(df_team['B365H'], df_team['B365A']), df_team['B365D'])
        all['B365Say'] = np.where(all['HomeMatch'],
                                  # home match
                                  np.where(all['B365Max'] == df_team['B365H'], -1,
                                           np.where(all['B365Max'] == df_team['B365A'], 1,
                                                    0)),
                                  # away match
                                  np.where(all['B365Max'] == df_team['B365H'], 1,
                                           np.where(all['B365Max'] == df_team['B365A'], -1,
                                                    0))
                                  )
        all['B365Diff'] = np.where(all['B365Say'] == 1, all['B365Max'] - all['B365Min'],
                                   all['B365Min'] - all['B365Max'])
        all['Corners'] = np.where(all['HomeMatch'], df_team['HC'], df_team['AC'])
        all['Shots'] = np.where(all['HomeMatch'], df_team['HS'], df_team['AS'])
        all['ShotsOnTarget'] = np.where(all['HomeMatch'], df_team['HST'], df_team['AST'])
        all['Points'] = np.where(all['Won'], 3,
                                 np.where(all['Draw'], 1, 0)
                                  )
        all['AdjustedPoints'] = np.where(all['HomeMatch'],
                                  # home match
                                         np.where(all['Won'], 1,
                                                  np.where(all['Draw'], 0, -1)
                                                  )
                                         ,
                                  # away match
                                         np.where(all['Won'], 1.5,
                                                  np.where(all['Draw'], 0.5, 0)
                                                  )
                                  )

        # find number of times won against this opponent in last 5 meetings
        for key, groupByOpponent in all.groupby('Opponent'):
            # keep index as new a column, will be restored and assigned back to X later
            idx = groupByOpponent.index

            # make match day an index because rolling need an index date
            xx = groupByOpponent.set_index('Date')
            xx['idx'] = idx
            # shift to exclude self
            xx['Last5AgainstThisOpponentWon'] = xx['Won'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
            xx['Last5AgainstThisOpponentDraw'] = xx['Draw'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
            # xx['Last5AgainstThisOpponentLost'] = xx['Lost'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)

            xx['Last3AgainstThisOpponentWon'] = xx['Won'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)
            xx['Last3AgainstThisOpponentDraw'] = xx['Draw'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)

            xx['LastAgainstThisOpponentWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
            xx['LastAgainstThisOpponentDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
            # xx['LastThisOpponentLost'] = xx['Lost'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)

            # restore index
            xx = xx.set_index('idx')

            # assign back to the big dataframe
            all.loc[xx.index, 'Last5AgainstThisOpponentWon'] = xx['Last5AgainstThisOpponentWon']
            all.loc[xx.index, 'Last5AgainstThisOpponentDraw'] = xx['Last5AgainstThisOpponentDraw']
            # X.loc[xx.index, 'Last5AgainstThisOpponentLost'] = xx['Last5AgainstThisOpponentLost']
            all.loc[xx.index, 'Last3AgainstThisOpponentWon'] = xx['Last3AgainstThisOpponentWon']
            all.loc[xx.index, 'Last3AgainstThisOpponentDraw'] = xx['Last3AgainstThisOpponentDraw']
            all.loc[xx.index, 'LastAgainstThisOpponentWon'] = xx['LastAgainstThisOpponentWon']
            all.loc[xx.index, 'LastAgainstThisOpponentDraw'] = xx['LastAgainstThisOpponentDraw']
            # X.loc[xx.index, 'LastThisOpponentLost'] = xx['LastThisOpponentLost']

        # stats by year/season
        for year, groupByYear in all.groupby('Year'):
            # print(year)
            # keep index as new a column, will be restored and assigned back to X later
            idx = groupByYear.index

            # make match day an index because rolling need an index date
            xx = groupByYear.set_index('Date')
            xx['idx'] = idx

            # shift to exclude self
            xx['CornersSoFar'] = np.nancumsum(xx['Corners'].shift())
            xx['ShotsSoFar'] = np.nancumsum(xx['Shots'].shift())
            xx['ShotsOnTargetSoFar'] = np.nancumsum(xx['ShotsOnTarget'].shift())

            xx['HomeWonNum'] = np.where(xx['HomeMatch'] & xx['Won'], 1, 0)
            xx['HomeWonSoFar'] = np.nancumsum(xx['HomeWonNum'].shift())
            xx['AwayWonNum'] = np.where((xx['HomeMatch'] == False) & xx['Won'], 1, 0)
            xx['AwayWonSoFar'] = np.nancumsum(xx['AwayWonNum'].shift())

            xx['PointsSoFar'] = np.nancumsum(xx['Points'].shift())
            xx['AdjustedPointsSoFar'] = np.nancumsum(xx['AdjustedPoints'].shift())

            # restore index
            xx = xx.set_index('idx')

            # assign back to the big dataframe
            # all.loc[xx.index, 'CornersSoFar'] = xx['CornersSoFar']
            # all.loc[xx.index, 'ShotsSoFar'] = xx['ShotsSoFar']
            # all.loc[xx.index, 'ShotsOnTargetSoFar'] = xx['ShotsOnTargetSoFar']
            # all.loc[xx.index, 'HomeWonSoFar'] = xx['HomeWonSoFar']
            # all.loc[xx.index, 'AwayWonSoFar'] = xx['AwayWonSoFar']
            all.loc[xx.index, 'PointsSoFar'] = xx['PointsSoFar']
            all.loc[xx.index, 'AdjustedPointsSoFar'] = xx['AdjustedPointsSoFar']

        # find recent forms
        idx = all.index
        xx = all.set_index('Date')
        xx['idx'] = idx
        xx['Last5Won'] = xx['Won'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['Last5Draw'] = xx['Draw'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
        # xx['Last5Lost'] = xx['Lost'].rolling(6).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['Last3Won'] = xx['Won'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['Last3Draw'] = xx['Draw'].rolling(4).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['LastWon'] = xx['Won'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)
        xx['LastDraw'] = xx['Draw'].rolling(2).apply(lambda x: np.nansum(x.shift()), raw=False)

        # restore index
        xx = xx.set_index('idx')
        # assign back to the big dataframe
        all.loc[xx.index, 'Last5Won'] = xx['Last5Won']
        all.loc[xx.index, 'Last5Draw'] = xx['Last5Draw']
        all.loc[xx.index, 'Last3Won'] = xx['Last3Won']
        all.loc[xx.index, 'Last3Draw'] = xx['Last3Draw']
        all.loc[xx.index, 'LastWon'] = xx['LastWon']
        all.loc[xx.index, 'LastDraw'] = xx['LastDraw']
        # X.loc[xx.index, 'Last5Lost'] = xx['Last5Lost']

        # replace nan with 0
        # TODO: better way to handle nan
        # all.loc[np.isnan(all['FTR']), 'FTR'] = ''
        all.loc[np.isnan(all['Last5AgainstThisOpponentWon']), 'Last5AgainstThisOpponentWon'] = 0
        all.loc[np.isnan(all['Last5AgainstThisOpponentDraw']), 'Last5AgainstThisOpponentDraw'] = 0
        # X.loc[np.isnan(X['Last5AgainstThisOpponentLost']), 'Last5AgainstThisOpponentLost'] = 0
        all.loc[np.isnan(all['Last3AgainstThisOpponentWon']), 'Last3AgainstThisOpponentWon'] = 0
        all.loc[np.isnan(all['Last3AgainstThisOpponentDraw']), 'Last3AgainstThisOpponentDraw'] = 0
        all.loc[np.isnan(all['LastAgainstThisOpponentWon']), 'LastAgainstThisOpponentWon'] = 0
        all.loc[np.isnan(all['LastAgainstThisOpponentDraw']), 'LastAgainstThisOpponentDraw'] = 0
        # X.loc[np.isnan(X['LastThisOpponentLost']), 'LastThisOpponentLost'] = 0
        all.loc[np.isnan(all['Last5Won']), 'Last5Won'] = 0
        all.loc[np.isnan(all['Last5Draw']), 'Last5Draw'] = 0
        # X.loc[np.isnan(X['Last5Lost']), 'Last5Lost'] = 0
        all.loc[np.isnan(all['Last3Won']), 'Last3Won'] = 0
        all.loc[np.isnan(all['Last3Draw']), 'Last3Draw'] = 0
        all.loc[np.isnan(all['LastWon']), 'LastWon'] = 0
        all.loc[np.isnan(all['LastDraw']), 'LastDraw'] = 0
        all.loc[np.isnan(all['B365Diff']), 'B365Diff'] = 0

        # restrict training data (too old data may not be irrelevance)
        X = all
        Y = X[['Result']]
        # del X['Result']
        # X = all.loc[(all['Year'] >= train_year) & (all['Year'] < predict_year)]
        # Y = all[['Result']]

        # split data into train - test sets
        # x_train = X[(X['Year'] < predict_year)]
        # y_train = Y[(X['Year'] < predict_year)]
        # x_test = X[(X['Year'] >= predict_year)]
        # y_test = Y[(X['Year'] >= predict_year)]
        # X['Predict'] = ''
        close_leaks(X)
        ret[team] = [X, Y]
    return ret


# call this after you've split data
def close_leaks(X):
    # remove duplicate features
    del X['LastWon']
    del X['LastDraw']

    # prevent future leaks
    # result = pd.DataFrame(X['Result'])
    del X['Result']
    del X['Lost']
    del X['Draw']
    del X['Won']
    del X['FTR']
    del X['Date']
    del X['Opponent']
    del X['Team']
    del X['B365Max']
    del X['B365Min']
    del X['Corners']
    del X['Shots']
    del X['ShotsOnTarget']
    del X['Points']
    del X['AdjustedPoints']
