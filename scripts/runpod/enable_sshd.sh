#!/bin/bash
set -euo pipefail

# scripts/runpod/enable_sshd.sh
# Bootstraps sshd on a Runpod instance through a basic SSH shell.

PUBLIC_KEY_CONTENT=${1:-""}

echo "--- Runpod SSH Bootstrap ---"

# 1. Detect environment
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container."
fi

# 2. Install openssh-server if missing
if ! command -v sshd >/dev/null 2>&1; then
    echo "sshd not found. Installing openssh-server..."
    apt-get update
    apt-get install -y openssh-server
else
    echo "sshd is already installed."
fi

# 3. Configure SSH directories and permissions
echo "Configuring SSH directories..."
mkdir -p /run/sshd
mkdir -p ~/.ssh
chmod 700 ~/.ssh

if [ -n "$PUBLIC_KEY_CONTENT" ]; then
    echo "Adding public key to authorized_keys..."
    echo "$PUBLIC_KEY_CONTENT" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    # Remove duplicates
    sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
else
    echo "No public key provided to authorized_keys."
fi

# 4. Generate host keys if missing
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    echo "Generating SSH host keys..."
    ssh-keygen -A
fi

# 5. Start sshd if not running
if ! pgrep -x sshd >/dev/null 2>&1; then
    echo "Starting sshd on port 22..."
    /usr/sbin/sshd
else
    echo "sshd is already running."
fi

# 6. Verify success
echo "Verification:"
if pgrep -x sshd >/dev/null 2>&1; then
    echo "[OK] sshd process is running."
else
    echo "[FAIL] sshd process not found."
    exit 1
fi

# Check if listening on port 22
if command -v ss >/dev/null 2>&1; then
    if ss -tlnp | grep -q :22; then
        echo "[OK] Listening on port 22."
    else
        echo "[WARN] Could not confirm port 22 with 'ss'."
    fi
elif command -v netstat >/dev/null 2>&1; then
    if netstat -tlnp | grep -q :22; then
        echo "[OK] Listening on port 22."
    else
        echo "[WARN] Could not confirm port 22 with 'netstat'."
    fi
else
    echo "Neither 'ss' nor 'netstat' found. Cannot verify port 22 directly."
fi

echo "--- Bootstrap Complete ---"
