## Importing libraries
import yfinance as yf
import pandas_datareader as pd
import datetime as dt
import numpy as np
from absl import flags, app
import os

## Setup
FLAGS = flags.FLAGS
flags.DEFINE_string("start", '2005-01-03', "Enter a start date: ")
flags.DEFINE_string("end", '2016-12-31', "Enter an end date: ")
flags.DEFINE_string("ticker", '^GSPC', "Enter a ticker symbol: ")
# flags.DEFINE_string("dataSet","/Users/tanvirsethi/Desktop/Almost all Docs/Georgia Institute of Technology-MSCS/GeorgiaTech/Spring'24/CS7641/Assignment1_SL/^GSPC.csv", "Input a dataset")

## To download data
def data_download(start,end,ticker):
    download_df = yf.download(ticker,start,end)
    save_loc = os.getcwd()

    try:
        TEMP = download_df.copy(deep=True)
        TEMP = TEMP.dropna()
        TEMP.to_csv(save_loc+"/"+ticker+".csv")
    except:
        print("Unable to load data for {}".format("SP500"))

    return


## To read the downloaded dataset and removig null values
# def readAndClean(dataSet):
#     df = pd.DataReader(dataSet)
#     df = df.dropna()

## Data Cleaning
# def cleaner(argv):
    
def main(argv):
    start = FLAGS.start
    end = FLAGS.end
    ticker = FLAGS.ticker
    # dataSet = FLAGS.dataSet
    dd = data_download(start,end,ticker)
    # if dd:
    #     dt = readAndClean(dataSet)
    # else:
    #     print("There is no data set to clean in this filepath.")
    return
    


if __name__ == '__main__':
    app.run(main)
