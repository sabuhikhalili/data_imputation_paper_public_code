import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from utilities import write_to_excel
import os

from datetime import datetime, timedelta

# get the lisst of S&P 500 companies from slickcharts.com
url = 'https://www.slickcharts.com/sp500'

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}
response = requests.get(url, headers=headers)
html_content = response.text

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Navigate to the specific section of the HTML tree where the table is located
# Replace 'div' with the appropriate HTML tag and attributes for your case
table_section = soup.find('div', {'class': 'table-responsive'})


# Parse the HTML content and extract tables into a list of DataFrames
table = pd.read_html(str(table_section))[0]

SnP500_companies = table.Symbol.to_list()


def fetch_data_in_batches(symbol, start_date, end_date, interval='1h'):
    batch_start = pd.to_datetime(start_date)
    batch_end = batch_start + timedelta(days=700)
    all_data = pd.DataFrame()

    while batch_start < pd.to_datetime(end_date):
        batch_end = min(batch_end, pd.to_datetime(end_date))

        data = yf.Ticker(symbol).history(start=batch_start, end=batch_end, interval=interval, keepna=True).Close
        all_data = pd.concat([all_data, data])

        batch_start = batch_end
        batch_end = batch_start + timedelta(days=730)

    return all_data


hist = pd.DataFrame()
for symbol in SnP500_companies:
    # use yahoo finance to get historical data for each stock
    equity =  yf.Ticker(symbol)
    hist[symbol] = hist[symbol] = fetch_data_in_batches(symbol, '1990-01-01', '2019-12-31')

# Drop columns with any NaN values
snp_balanced = hist.dropna(axis=1)

# Convert 'Date' column to timezone-unaware datetime
snp_balanced.index = snp_balanced.index.tz_convert(None)

print(snp_balanced.shape)

write_to_excel(snp_balanced, filepath=os.path.join(data_dir, 'snp_balanced.xlsx'))
