# Set operating system - possibly not needed on lab workstation
FROM ubuntu:18.04

# Install
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y python3.7 \
    python3-pip \
    graphviz
RUN pip3 install --upgrade pip

# Set wd
WORKDIR /root
COPY . .

# Install dependencies
RUN pip install -r /root/requirements.txt

# Make all scripts executable
RUN find . -type f -iname "*.py" -exec chmod +x {} \;

# Add to environment path
ENV PATH "/root:$PATH"