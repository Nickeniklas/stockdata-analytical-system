import streamlit as st  # Importing the Streamlit library for creating web apps
import plotly.express as px  # Importing Plotly Express for creating plots
import pandas as pd  # Importing pandas for data manipulation

# Reading the CSV file into a DataFrame
df = pd.read_csv("output/predictions/prophet_predictions.csv", header=0, encoding='unicode_escape')

# Page setup
st.set_page_config(page_title="Visualization of Apple stock - Yahoo Finance", page_icon=":bar_chart:", layout="wide")  # Setting the page configuration

st.title(" :bar_chart: Apple Stock Visualization")  # Setting the title of the web page
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)  # Adding custom CSS to the page

st.write("Analytical System Design 2025 - Final Project") 
st.write("The following line chart shows the historical stock prices of Apple Inc. (AAPL) from this day three years back.")  
st.write("Predictions made by the Prophet model for the Apple stock.")

# Displaying the DataFrame
st.write(df)  

st.line_chart(df['yhat']) # Creating a line chart using Plotly Express