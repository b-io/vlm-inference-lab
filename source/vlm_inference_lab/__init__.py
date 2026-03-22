import os

def load_env(file_path=".env"):
    """Loads environment variables from a file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                # Split the line into key and value
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

def get_timezone():
    """Returns the default timezone from the environment."""
    return os.getenv("TIMEZONE", "Europe/Zurich")

# Load environment variables from .env if present
load_env()
