from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    question: str
    answer: str

class QuestionRequest(BaseModel):
    pdf_name: str
    question: str
    model: str = "deepseek-v2:16b"
    history: Optional[List[ChatMessage]] = None
