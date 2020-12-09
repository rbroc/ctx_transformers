# Set operating system - possibly not needed on lab workstation
FROM ubuntu:18.04

# Install
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y python3.7
RUN pip install --upgrade pip

# Set wd
WORKDIR /root
COPY . .

# pip install from requirements list
RUN pip install -r /root/requirements.txt

# Make all scripts executable
RUN find . -type f -iname "*.py" -exec chmod +x {} \; 