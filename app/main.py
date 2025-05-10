from fastapi import FastAPI
import uvicorn
from app.routes.upload import router as upload_router
from app.routes.question import router as question_router
from app.services.embedding_service import initialize_embedding_model, cleanup_embedding_model, get_embedding_model
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="PDF QA API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """
    Initialize resources when the application starts
    """
    try:
        initialize_embedding_model()
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources when the application shuts down
    """
    logger.info("👋 Cleaning up resources...")
    cleanup_embedding_model()
    logger.info("✅ Cleanup completed")

# health check
@app.get("/health", tags=["health"], status_code=200)
async def health_check():
    logger.info("🔍 Health check endpoint called")
    try:
        model = get_embedding_model()
        model_status = model is not None
    except Exception:
        model_status = False
    
    return {
        "status": "healthy",
        "message": "🚀 Server is up and running!",
        "version": "1.0.0",
        "embedding_model_loaded": model_status,
        "worker_id": os.getpid()
    }

app.include_router(upload_router, prefix="/api/v1")
app.include_router(question_router, prefix="/api/v1")

if __name__ == "__main__":
    # Run the server with the specified host and port
    logger.info("🚀 Starting server on port 8080")
    logger.info("📚 API Documentation available at http://localhost:8080/docs")
    logger.info("💪 Health check endpoint available at http://localhost:8080/health")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)