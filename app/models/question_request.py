from pydantic import BaseModel

class QuestionRequest(BaseModel):
    pdf_name: str
    question: str
