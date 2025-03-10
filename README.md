# Stockdata Analytical System

## Tech stack
**Docker, Python, yFinance, PySpark, Prophet, Streamlit** 

## Project description
Get data from Yahoo Finance, with yfinance python module.

Process and analyze data with Spark - PySpark.

Make Predictions/forecasting with Meta’s Prophet.

Visualize in the browser with Streamlit.


## Setup
1. Clone repository:

    ```
    git clone https://github.com/Nickeniklas/stockdata-analytical-system.git
    ```

2. Go to docker folder:
   ```
   cd stockdata-analytical-system
   cd docker
   ```
3. Building and starting the application:
   ```
   docker-compose up --build
   ```

**This might take a while, so be patient.**

Stock data and Predictions data in `/docker/output` folder

Streamlit links for dynamic visualization should appear in console.
