import os
import requests

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
    
    print('done')


if __name__ == '__main__':
    main()
