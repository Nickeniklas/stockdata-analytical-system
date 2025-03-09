import pyspark
from pyspark.sql import SparkSession
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# first fetch data from yahoo finance, 
# then handle data with Spark, 
# and finally export data as csv file

# define the stock we want to fetch data for, and the period of time
stock_ticker = "AAPL"
period = "3y"

# get defined years of defined stock data
yf_data = yf.download(stock_ticker, period=period)

# date into column
yf_data.reset_index(inplace=True)

# better column names
# Rename columns
yf_data.columns = [
    'Date', 'Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Open_AAPL', 'Volume_AAPL'
]

# initialize spark session
spark = SparkSession.builder.appName("StockData").getOrCreate()

# create dataframe
df = spark.createDataFrame(yf_data)

# Handle missing values and outliers
df = df.na.drop() \
    .filter((col("Close_AAPL") > 0) & (col("High_AAPL") > 0))

# Create a column "priceAboveYesterday" with (1) for price above yesterday closing price (0) for under.
df = df.withColumn(
    "priceAboveYesterday",
    F.when(F.col("Close_AAPL") > F.lag(F.col("Close_AAPL"), 1).over(Window.orderBy("Date")), 1).otherwise(0)
)

# time based features (columns for: year, month, day)
df = df.withColumn(
    "year", F.year(F.col("Date"))
).withColumn(
    "month", F.month(F.col("Date"))
    ).withColumn(
    "day", F.dayofmonth(F.col("Date"))
)

# create column for price to volume ratio
df = df.withColumn(
    "priceToVolume",
    F.col("Close_AAPL") / F.col("Volume_AAPL")
)

#previous day price
df = df.withColumn(
    "Close_price_lag_1",
    F.lag(F.col("Close_AAPL"), 1).over(Window.orderBy("Date"))
)
#5 day moving average
df = df.withColumn(
  "Close_avg_5_days",
    F.avg(F.col("Close_AAPL")).over(Window.orderBy("Date").rowsBetween(-4, 0)),
)
#30 days moving average
df = df.withColumn(
    "Close_avg_30_days",
    F.avg(F.col("Close_AAPL")).over(Window.orderBy("Date").rowsBetween(-29, 0))
)
#7 day volatility
df = df.withColumn(
    "volatility_7_days",
    F.stddev(F.col("Close_AAPL")).over(Window.orderBy("Date").rowsBetween(-6, 0))
)
#30 day rolling std
df = df.withColumn(
    "volatility_30_days",
    F.stddev(F.col("Close_AAPL")).over(Window.orderBy("Date").rowsBetween(-29, 0))
)
# remove "nulls"
df = df.fillna({"Close_price_Lag_1": 0, "volatility_7_days": 0, "volatility_30_days": 0})

print("Dataframe schema:")
df.printSchema()

# Save the dataframe as a CSV file
df.coalesce(1).write.mode("overwrite").csv("/app/output/stock_data", header=True)

spark.stop()