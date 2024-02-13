## Importing libraries
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from absl import flags, app
import os

## Setup
FLAGS = flags.FLAGS
flags.DEFINE_string("start", '2005-01-03', "Enter a start date: ")
flags.DEFINE_string("end", '2016-12-31', "Enter an end date: ")
flags.DEFINE_string("ticker", '^GSPC', "Enter a ticker symbol: ")
flags.DEFINE_string("location", os.getcwd(), "Enter your preferred output filepath: ")


## To download data
def data_download(start, end, ticker, save_loc):
    download_df = yf.download(ticker, start, end)

    try:
        download_df.to_csv(os.path.join(save_loc, f"{ticker}.csv"))  # Save directly without creating a copy
    except Exception as e:
        print(f"Unable to save data for {ticker}. Error: {e}")

def cleanup_date_and_save(ticker, save_loc):
    df = pd.read_csv(os.path.join(save_loc, f"{ticker}.csv"))
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' to datetime format
    df['Date'] = df['Date'].dt.strftime('%Y%m%d').astype(int)  # Format and convert to int

    # Save cleaned DataFrame to a new CSV file
    cleaned_csv_path = os.path.join(save_loc, f"cleaned_{ticker}.csv")
    df.to_csv(cleaned_csv_path, index=False)

    return cleaned_csv_path

def main(argv):
    start = FLAGS.start
    end = FLAGS.end
    ticker = FLAGS.ticker
    save_loc = FLAGS.location

    data_download(start, end, ticker, save_loc)

    cleaned_csv_path = cleanup_date_and_save(ticker, save_loc)

    print(f"Cleaned DataFrame saved to: {cleaned_csv_path}")

if __name__ == '__main__':
    app.run(main)
