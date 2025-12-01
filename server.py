from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import run_rag

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class RAGQuery(BaseModel):
    query: str

@app.get("/")
def serve_ui():
    return FileResponse("frontend/rag_ui.html")

@app.post("/rag/query")
def rag_query(request: RAGQuery):
    result = run_rag(request.query)

    return {
        "query": request.query,
        "answer": result["answer"],
        "sources": result["sources"],
        "context": result["context"]
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
