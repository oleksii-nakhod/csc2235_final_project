#!/bin/bash

# Test if script is working
echo "Setup script is running..." >> /local/repository/setup.log

# Wait for any automatic system updates to finish
sudo DEBIAN_FRONTEND=noninteractive apt-get -y update

# Install pip
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip

# Install your project's dependencies
# As per section 4.4, the repo is cloned to /local/repository
pip3 install -r /local/repository/requirements.txt