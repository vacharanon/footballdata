import os
import requests
from shutil import copyfile

seasons = {
    2019: '1920',
    2018: '1819',
    2017: '1718',
    # ----
    2016: '1617',
    2015: '1516',
    2014: '1415',
    2013: '1314',
    2012: '1213',
    2011: '1112',
    2010: '1011',
    2009: '0910',
    2008: '0809',
    2007: '0708',
    2006: '0607',
    2005: '0506',
    # ----
    2004: '0405',
    2003: '0304',
    2002: '0203',
    2001: '0102',
    2000: '0001',
    1999: '9900',
    1998: '9899',
    1997: '9798',
    1996: '9697',
    1995: '9596',
    1994: '9495',
    1993: '9394',
}
leagues = {
    'E0': {
        'name': 'english',  # English Premier League full stats from 03/04 onward
        'full': 2003,
        'first': 1993,
    },
    'SC0': {
        'name': 'italian',  # Scottish Premier League full stats from 03/04 onward
        'full': 2003,
        'first': 1994,
    },
    'D1': {
        'name': 'scottish',  # German Bundesliga 1 full stats from 05/06 onward
        'full': 2005,
        'first': 1993,
    },
    'I1': {
        'name': 'italian',  # Italian Calcio Serie A full stats from 05/06 onward
        'full': 2005,
        'first': 1993,
    },
    'SP1': {
        'name': 'spanish',  # Spanish La Liga full stats from 05/06 onward
        'full': 2005,
        'first': 1993,
    },
    'F1': {
        'name': 'french',  # French Le Championnat full stats from 05/06 onward
        'full': 2005,
        'first': 1993,
    },
    'N1': {
        'name': 'dutch',  # Dutch Eredivisie full stats from 17/18 onward
        'full': 2017,
        'first': 1993,
    },
    'B1': {
        'name': 'belgian',  # Belgian Jupiler League full stats from 17/18 onward
        'full': 2017,
        'first': 1995,
    },
    'P1': {
        'name': 'portuguese',  # Portuguese Liga I full stats from 17/18 onward
        'full': 2017,
        'first': 1994,
    },
    'T1': {
        'name': 'turkish',  # Turkish Futbol Ligi 1 full stats from 17/18 onward
        'full': 2017,
        'first': 1994,
    },
    'G1': {
        'name': 'greek',  # Greek Ethniki Katigoria full stats from 17/18 onward
        'full': 2017,
        'first': 1994,
    },
}


def main():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('fulldata'):
        os.makedirs('fulldata')
    for (year, season) in seasons.items():
        for (league, league_info) in leagues.items():
            if year < league_info['first']:
                continue
            print('downloading %s %s' % (season, league_info['name']))

            url = 'http://www.football-data.co.uk/mmz4281/%s/%s.csv' % (season, league)

            response = requests.get(url, stream=True)

            if not os.path.exists('data/%s' % league_info['name']):
                os.makedirs('data/%s' % league_info['name'])

            dest1 = 'data/%s/%s.csv' % (league_info['name'], year)
            with open(dest1, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

            if year >= league_info['full']:
                dest2 = 'fulldata/%s/%s.csv' % (league_info['name'], year)
                if not os.path.exists('fulldata/%s' % league_info['name']):
                    os.makedirs('fulldata/%s' % league_info['name'])
                copyfile(dest1, dest2)


    print('done')


if __name__ == '__main__':
    main()
