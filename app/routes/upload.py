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
    filename: str


@router.post("/upload_pdf/")
async def upload_pdf(request: UploadRequest):
    """
    Upload PDF and create embeddings
    """

    try:
        filepath = request.filepath
        file_name = request.filename

        # Handle both remote URLs and local file paths
        if filepath.startswith("http://") or filepath.startswith("https://"):
            # For remote URLs
            response = requests.get(filepath)
            response.raise_for_status()

            pdf_path = os.path.join("./uploads", file_name)
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        elif filepath.startswith("file://"):
            # For local file:// URLs
            local_path = filepath.replace("file://", "")
            file_name = os.path.basename(local_path)
            pdf_path = os.path.join("./uploads", file_name)

            # If the file is already in the uploads directory, just use it directly
            if os.path.exists(local_path):
                # Copy the file to uploads directory if it's not already there
                if local_path != pdf_path:
                    shutil.copy2(local_path, pdf_path)
            else:
                raise HTTPException(
                    status_code=404, detail=f"File not found: {local_path}"
                )
        else:
            # Assume it's a regular local path
            file_name = os.path.basename(filepath)
            pdf_path = os.path.join("./uploads", file_name)

            if os.path.exists(filepath):
                # Copy the file to uploads directory if it's not already there
                if filepath != pdf_path:
                    shutil.copy2(filepath, pdf_path)
            else:
                raise HTTPException(
                    status_code=404, detail=f"File not found: {filepath}"
                )

        # Process PDF to create embeddings
        process_pdf(pdf_path, file_name)

        return {"message": "PDF processed successfully", "pdf_name": file_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
