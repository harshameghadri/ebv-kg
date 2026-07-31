"""Main FastAPI application entrypoint for the EBV Knowledge System."""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router

app = FastAPI(
    title="EBV Knowledge System API",
    description="RAG and Knowledge Graph system for Epstein-Barr Virus research",
    version="0.1.0",
)

# Register API routes
app.include_router(api_router, prefix="/api")

# Mount app/static directory at root / to serve the frontend dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
