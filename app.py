from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import get_response

app = FastAPI()



# Request format
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question

    answer = get_response(query)
    return answer

