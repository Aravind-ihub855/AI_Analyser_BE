from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QuestionRequest(BaseModel):
    question: str
    dateFilter: Optional[str] = None
    chat_context: Optional[List[str]] = []

class SummarizeRequest(BaseModel):
    messages: List[str]

class ChatResponse(BaseModel):
    answer: str
    tableData: Optional[Dict[str, Any]] = None
    chartData: Optional[str] = None 
    reportData: Optional[str] = None
    dashboardData: Optional[Dict[str, Any]] = None
    voice_enabled: bool = True
