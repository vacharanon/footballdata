from bs4 import BeautifulSoup
import requests
from datetime import timedelta, date
import pandas as pd
import traceback


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def main():
    country = 'uk'
    cities = [
        # 'manchester',
        'london',
        'bournemouth',
        'brighton',
        'burnley',
        'liverpool',
        'huddersfield',
        'leicester',
        'newcastle-upon-tyne',
        'southampton',
        'stoke-on-trent',
        'swansea',
        'watford',
        'west-bromwich',
        'watford',
    ]
    start_date = date(2011, 8, 1)
    end_date = date(2018, 5, 30)

    aggr = {
        'date': [],
        'temp_high': [],
        'temp_low': [],
        'temp_mean': [],
        'wind_speed': [],
        'humidity_high': [],
        'humidity_low': [],
        'humidity_mean': [],
        'is_rain': []
    }
    for city in cities:
        print(f'{country}/{city}')
        try:
            for single_date in daterange(start_date, end_date):
                temps = []
                weathers = []
                wind_speeds = []
                wind_directions = []
                humidities = []

                ymd = single_date.strftime("%Y%m%d")
                print(ymd)
                uri = f'https://www.timeanddate.com/scripts/cityajax.php?n={country}/{city}&mode=historic&hd={ymd}&month={single_date.month}&year={single_date.year}'
                r = requests.get(uri)
                data = r.text
                doc = BeautifulSoup(data, 'lxml')

                # history_table = doc.find('table', {'id': 'wt-his'}).find('tbody')
                history_table = doc.find('tbody')
                for tr in history_table.find_all('tr'):
                    cells = tr.find_all('td')
                    if len(cells) < 2:
                        continue
                    if cells[1].get_text() == 'N/A':
                        continue
                    if cells[5].get_text() == 'N/A':
                        continue
                    # weather_icon = cells[0].get_text()
                    temp = int(cells[1].get_text().strip(' °C'))
                    temps.append(temp)

                    weather = cells[2].get_text()
                    weathers.append(weather)

                    wind_speed = 0 if (cells[3].get_text() == 'No wind' or cells[3].get_text() == 'N/A') else int(cells[3].get_text().strip(' km/h'))
                    wind_speeds.append(wind_speed)

                    wind_direction = cells[4].span['title']
                    wind_directions.append(wind_direction)

                    humidity = int(cells[5].get_text().strip('%'))  # percent
                    humidities.append(humidity)
                    # barometer = cells[6].get_text()
                    # visibility = cells[7].get_text()

                if len(temps) < 1:
                    continue
                df_single_date = pd.DataFrame(
                    data={
                        'date': single_date,
                        'temp': temps,
                        'weather': weathers,
                        'wind_speed': wind_speeds,
                        'wind_direction': wind_directions,
                        'humidity': humidities
                    }
                )

                aggr['date'].append(single_date)
                aggr['temp_high'].append(df_single_date['temp'].max())
                aggr['temp_low'].append(df_single_date['temp'].min())
                aggr['temp_mean'].append(int(df_single_date['temp'].mean()))
                aggr['wind_speed'].append(single_date)
                aggr['humidity_high'].append(df_single_date['humidity'].max())
                aggr['humidity_low'].append(df_single_date['humidity'].min())
                aggr['humidity_mean'].append(int(df_single_date['humidity'].mean()))
                gg = df_single_date[
                    (df_single_date['weather'].str.contains('rain')) |
                    df_single_date['weather'].str.contains('shower') |
                    df_single_date['weather'].str.contains('sprinkle') |
                    df_single_date['weather'].str.contains('Sprinkle')]
                aggr['is_rain'].append(len(gg) > 1)
        except:
            traceback.print_exc()
        finally:
            df_aggr = pd.DataFrame(
                data=aggr,
            )
            df_aggr.set_index('date', inplace=True)
            df_aggr.to_csv(f'weather_{country}_{city}.csv')


if __name__ == '__main__':
    main()
