from fastapi import FastAPI

app = FastAPI(
    title="EBV Knowledge System API",
    description="RAG and Knowledge Graph system for Epstein-Barr Virus research",
    version="0.1.0",
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the EBV Knowledge System API"}
