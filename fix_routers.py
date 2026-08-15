import os

main_code = '''"""Main FastAPI application entrypoint for the EBV Knowledge System."""

import os
from dotenv import load_dotenv
load_dotenv()

if os.getenv(HF_TOKEN):
    os.environ[HF_TOKEN] = os.getenv(HF_TOKEN).strip()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.api.hypothesis_routes import router as hypothesis_router
from app.api.health_routes import router as health_router
from app.api.auth_routes import router as auth_router

app = FastAPI(
    title=EBV Knowledge System API,
    description=RAG and Knowledge Graph system for Epstein-Barr Virus research,
    version=0.1.0,
)

# Register API routers with /api prefix and root
app.include_router(api_router, prefix=/api)
app.include_router(hypothesis_router, prefix=/api)
app.include_router(health_router, prefix=/api)
app.include_router(auth_router, prefix=/api)

app.include_router(api_router)
app.include_router(hypothesis_router)
app.include_router(health_router)
app.include_router(auth_router)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), static)
os.makedirs(static_dir, exist_ok=True)
app.mount(/, StaticFiles(directory=static_dir, html=True), name=static)
'''

with open(/storage/harsha_projects/ebv_KG/app/main.py, w) as f:
    f.write(main_code)

# 2. Update health_routes.py
health_code = '''"""FastAPI Health and Metrics Router for the EBV Knowledge System."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.vector import LanceDBClient
from app.materialization.neo4j_client import Neo4jClient
from app.materialization.kuzu_engine import KuzuEngine
from app.api.routes import get_pg_conn, get_neo4j_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Health
