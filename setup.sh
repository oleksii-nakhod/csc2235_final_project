#!/bin/bash
echo "Setup script is running as $(whoami)..." >> /local/repository/setup.log

sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-venv python3-pip

# 2. Find the non-root user who owns the project directory
USER_NAME=$(stat -c '%U' /local/repository)
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)

if [ -z "$USER_NAME" ] || [ -z "$USER_HOME" ]; then
    echo "Could not find project owner or home directory. Exiting." >> /local/repository/setup.log
    exit 1
fi

echo "Found project user: $USER_NAME with home: $USER_HOME" >> /local/repository/setup.log

# 3. Create and activate a virtual environment
sudo -u "$USER_NAME" bash -c "
    echo 'Creating venv as $USER_NAME...' >> /local/repository/setup.log
    python3 -m venv /local/repository/venv
    
    echo 'Activating venv and installing requirements...' >> /local/repository/setup.log
    source /local/repository/venv/bin/activate
    pip install --upgrade pip
    pip install -r /local/repository/requirements.txt
"

# Install dependencies inside the venv
pip install --upgrade pip
pip install -r /local/repository/requirements.txt

# 4. Add venv activation to the user's .profile (for SSH login)
PROFILE_PATH="$USER_HOME/.profile"
ACTIVATE_LINE="source /local/repository/venv/bin/activate"

# Check if the line already exists to avoid duplicates
if ! grep -qF -- "$ACTIVATE_LINE" "$PROFILE_PATH"; then
    echo "Adding venv activation to $PROFILE_PATH" >> /local/repository/setup.log
    echo "" >> "$PROFILE_PATH"
    echo "# Automatically activate project venv" >> "$PROFILE_PATH"
    echo "$ACTIVATE_LINE" >> "$PROFILE_PATH"
else
    echo "Venv activation already in $PROFILE_PATH" >> /local/repository/setup.log
fi

echo "Setup complete. venv will auto-activate on next login for $USER_NAME." >> /local/repository/setup.log