from fastapi import FastAPI
import uvicorn
from app.routes.upload import router as upload_router
from app.routes.question import router as question_router
import os
import logging

# Create FastAPI app
app = FastAPI(title="PDF QA API", version="1.0.0")

# Include routes
app.include_router(upload_router, prefix="/api/v1")
app.include_router(question_router, prefix="/api/v1")

if __name__ == "__main__":
    # Run the server with the specified host and port
    uvicorn.run("app.main:app")