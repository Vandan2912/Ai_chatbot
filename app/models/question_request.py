from pydantic import BaseModel

class QuestionRequest(BaseModel):
    pdf_name: str
    question: str
    model: str = "deepseek-v2:16b"
