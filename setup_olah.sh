#!/bin/bash

# -------------------------------------------------------------------------
# FILE: setup-olah.sh
# ROLE: Local AI Model Mirror & Storage Orchestrator
# Execution: To be executed on the VM after the ZFS script is executed on PVE.
#
# DESCRIPTION:
# Automates the deployment of the Olah HuggingFace mirror. Handles 
# virtual environment creation, ZFS-backed cache directory initialization, 
# and the generation of the systemd service unit. Ensures the local 
# API is accessible for the AI video editing pipeline.
#
# HARDWARE & STORAGE COMPATIBILITY:
# - Optimized for ZFS pools (FastScratch) for high-speed model I/O.
# - Configures systemd to manage local port 8090 binding.
# - Integrates with local NVIDIA GPU workflows via local model hosting.
# -------------------------------------------------------------------------



# Configuration
STORAGE_PATH="/mnt/FastScratch/olah-cache"
INSTALL_DIR="/opt/olah"
SERVICE_FILE="/etc/systemd/system/olah.service"
MOUNT_TAG="models" # This must match the tag in your 104.conf

echo "--- Starting Idempotent Olah Setup (VM Side) ---"

# 1. Prepare Mount Point
sudo mkdir -p "$STORAGE_PATH"

# 2. Idempotent Mount Logic
# Check if already mounted; if not, mount it.
if ! mountpoint -q "$STORAGE_PATH"; then
    echo "Mounting 9p share from host..."
    sudo mount -t 9p -o trans=virtio,version=9p2000.L,rw "$MOUNT_TAG" "$STORAGE_PATH"
fi

# 3. Idempotent fstab update
# Only add the line if the tag 'models' isn't already in /etc/fstab
if ! grep -q "$MOUNT_TAG" /etc/fstab; then
    echo "Adding mount to /etc/fstab..."
    echo "$MOUNT_TAG $STORAGE_PATH 9p trans=virtio,version=9p2000.L,rw,nofail 0 0" | sudo tee -a /etc/fstab
else
    echo "fstab entry already exists. Skipping."
fi

# 4. Verify the mount actually works before proceeding
if [ ! -w "$STORAGE_PATH" ] && [ "$(id -u)" != "0" ]; then
    # If not writable by user, try to fix permissions once
    sudo chown "$USER":"$USER" "$STORAGE_PATH"
fi

# 5. Setup / Update Virtual Environment
sudo mkdir -p "$INSTALL_DIR"
sudo chown -R "$USER":"$USER" "$INSTALL_DIR"

if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$INSTALL_DIR/venv"
fi

echo "Syncing Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --upgrade olah

# 6. Idempotent Systemd Service Creation
TEMP_SERVICE=$(mktemp)
cat <<EOF > "$TEMP_SERVICE"
[Unit]
Description=Olah HuggingFace Local Mirror
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/opt/olah
Environment="CACHE_DIR=/mnt/FastScratch/olah-cache"
Environment="CACHE_SIZE_LIMIT=500GB"
Environment="CACHE_CLEAN_STRATEGY=LRU"

ExecStart=/opt/olah/venv/bin/olah-cli --port 8090 --host 0.0.0.0 --repos-path /mnt/FastScratch/olah-cache --cache-size-limit 500GB --cache-clean-strategy LRU

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

if ! diff -q "$TEMP_SERVICE" "$SERVICE_FILE" > /dev/null 2>&1; then
    echo "Updating systemd service file..."
    sudo cp "$TEMP_SERVICE" "$SERVICE_FILE"
    sudo systemctl daemon-reload
else
    echo "Service file is already up to date."
fi
rm "$TEMP_SERVICE"

# 7. Ensure service is enabled and running
echo "Ensuring service is active..."
sudo systemctl enable olah
sudo systemctl restart olah

# 8 If there are other VMs accessing Olah service and  UFW (Uncomplicated Firewall) is in use
#sudo ufw allow 8090/tcp

echo "--- Setup Complete ---"
systemctl status olah --no-pager