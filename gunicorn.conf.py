import os

# Get port from environment or fallback to 10000
port = os.environ.get("PORT", 10000)
bind = f"0.0.0.0:{port}"

# Limit workers to 1 to prevent Out-Of-Memory (OOM) errors on Render's 512MB free tier
workers = 1

# Threads help handle concurrent requests within the single worker
threads = 4

# High timeout (120s) because PyTorch model loading and inference might take time
timeout = 120
