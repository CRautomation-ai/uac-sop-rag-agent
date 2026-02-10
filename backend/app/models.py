from pydantic import BaseModel
from typing import List, Optional


class PreviousMessage(BaseModel):
    query: str
    answer: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    previous_messages: Optional[List[PreviousMessage]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class LoadDocumentsResponse(BaseModel):
    message: str
    chunks_processed: int
    files_processed: int


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    documents_loaded: bool


class AuthRequest(BaseModel):
    password: str


class AuthResponse(BaseModel):
    token: str
