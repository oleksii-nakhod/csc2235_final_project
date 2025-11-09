#!/bin/bash
echo "Setup script is running..." >> /local/repository/setup.log

sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-venv python3-pip

# Create and activate a virtual environment
python3 -m venv /local/repository/venv
source /local/repository/venv/bin/activate

# Install dependencies inside the venv
pip install --upgrade pip
pip install -r /local/repository/requirements.txt