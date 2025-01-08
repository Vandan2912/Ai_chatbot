from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_processing import process_pdf
import os
import shutil

router = APIRouter()

@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and create embeddings
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Save uploaded PDF
        pdf_path = os.path.join("./uploads", file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF to create embeddings
        process_pdf(pdf_path, file.filename)

        return {"message": "PDF processed successfully", "pdf_name": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
