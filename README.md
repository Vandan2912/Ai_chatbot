

Here's a clean and well-structured `README.md` file for your AI chatbot:  

```markdown
# AI Chatbot

This repository contains the source code for an AI chatbot built using FastAPI. Follow the steps below to set up and run the application locally or on a remote server.


## Requirements
- Python 3.10 or higher
- pip (Python package manager)

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
For Linux/macOS:
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Start the Server

* main:app: Specifies the FastAPI application instance (app) in the main.py file.

* --workers 4: Sets the number of worker processes to 4 (adjust as needed). A common recommendation is (2 x $num_cores) + 1.

* --worker-class uvicorn.workers.UvicornWorker: Specifies the worker class to use, which enables Gunicorn to work with ASGI applications.

* --bind 0.0.0.0:8000: Binds the application to all interfaces on port 8000.

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
```

```bash
gunicorn app.main:app -c gunicorn_config.py
```

### 2. Access the API Documentation
- **For Local Testing:** [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)  
- **For Remote AWS Server:** [http://18.210.142.6:8080/docs](http://18.210.142.6:8080/docs)

Attach LLm Screen :- screen -r 1118.pts-0.ip-172-31-34-64
and for detach :- Ctrl+A, D.

# Available Modes is 
# 1. llama3.1:8b
# 2. deepseek-v2:16b