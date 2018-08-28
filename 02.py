import os
import pandas as pd

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
df = df[['Year', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
ath_matrid = df.loc[(df['HomeTeam'] == 'Ath Madrid') | (df['AwayTeam'] == 'Ath Madrid')]
X = pd.DataFrame(
    data={
        'Year': ath_matrid['Year'],
        'HomeMatch': ath_matrid['HomeTeam'] == 'Ath Madrid',
    }
)
X['HalfTimeGoals'] = (ath_matrid['HTHG'] if X['HomeMatch'] is True else ath_matrid['HTAG'])
X['HalfTimeOpponentGoals'] = (ath_matrid['HTAG'] if X['HomeMatch'] is True else ath_matrid['HTHG'])
X['HalfTimeLead'] = X['HalfTimeGoals'] > X['HalfTimeOpponentGoals']
X['HalfTimeLeadMoreThanTwo'] = (X['HalfTimeGoals'] - X['HalfTimeOpponentGoals']) > 2
Y = ath_matrid['FTR']
# df5yrs = df.loc[(df['Year'] >= 2012) & (df['Year'] <= 2016)]
# print(len(df5yrs))
