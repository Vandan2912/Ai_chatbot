from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_processing import process_pdf
import os
import shutil
import requests
from pydantic import BaseModel

router = APIRouter()

# @router.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     """
#     Upload PDF and create embeddings
#     """
#     if not file.filename.endswith(".pdf"):
#         print(file.filename)
#         raise HTTPException(status_code=400, detail="File must be a PDF")
    
#     try:
#         # Save uploaded PDF
#         pdf_path = os.path.join("./uploads", file.filename)
#         with open(pdf_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Process PDF to create embeddings
#         process_pdf(pdf_path, file.filename)

#         return {"message": "PDF processed successfully", "pdf_name": file.filename}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

class UploadRequest(BaseModel):
    filepath: str

@router.post("/upload_pdf/")
async def upload_pdf(request: UploadRequest):
    """
    Upload PDF and create embeddings
    """

    try:
        filepath = request.filepath
        response = requests.get(filepath)
        response.raise_for_status()

        file_name = filepath.split("/")[-1]
        pdf_path = os.path.join("./uploads", file_name)
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        # get a file name in the path
        file_name = filepath.split("/")[-1]
        # Save uploaded PDF

        # Process PDF to create embeddings
        process_pdf(pdf_path, file_name)

        return {"message": "PDF processed successfully", "pdf_name": file_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
