# gunicorn_config.py

# Number of worker processes
workers = 4

# Host and port to bind to
bind = "0.0.0.0:8080"

# Worker class to use
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout for worker processes
timeout = 120

# Enable logging
accesslog = "access.log"
errorlog = "error.log"
loglevel = "info"