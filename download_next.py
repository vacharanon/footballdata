import os
import requests
from shutil import copyfile

from download import seasons, leagues


def main():
    if not os.path.exists('next'):
        os.makedirs('next')
    url = 'http://www.football-data.co.uk/fixtures.csv'
    response = requests.get(url, stream=True)
    dest1 = 'next/fixtures.csv'
    with open(dest1, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    year = 2018
    season = seasons[year]
    league = 'E0'
    league_info = leagues[league]
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
