from fastapi import APIRouter, HTTPException
from app.models.question_request import QuestionRequest
from app.services.qa_service import get_answer
import os

router = APIRouter()

@router.post("/ask_question/")
async def ask_question(request: QuestionRequest):
    """
    Answer questions about a processed PDF
    """

    vector_store_path = os.path.join("./vector_store", f"{request.pdf_name}_vectors")

    # Check if vector store exists
    if not os.path.exists(vector_store_path):
        raise HTTPException(status_code=404, detail="PDF not found. Please upload it first.")

    try:
        # Get answer using the QA service
        answer = get_answer(request.question, vector_store_path)
        return {"question": request.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
