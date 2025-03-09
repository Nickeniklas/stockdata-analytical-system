# Use an OpenJDK base image
FROM openjdk:11-jre-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt . 

# Install Python and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    gnupg \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

# pip installs
#RUN pip3 install pyspark yfinance pandas numpy matplotlib prophet holidays mlflow streamlit plotly
RUN pip3 install -r requirements.txt

# Copy the local app code to the container
COPY . /app

