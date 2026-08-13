"""FastAPI router defining authentication and reviewer session routes."""

import os
import uuid
import hashlib
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Helpers ---

def get_pg_conn():
    pg_dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN") or "postgresql://postgres:postgrespassword@localhost:5432/ebv_rag"
    conn = psycopg.connect(pg_dsn)
    try:
        yield conn
    finally:
        conn.close()

def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Hash password using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return f"{salt.hex()}:${key.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$key hash."""
    try:
        salt_hex, key_hex = stored_hash.split('$')
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return secrets.compare_digest(key, expected_key)
    except Exception:
        return False

def ensure_tables_and_seed_users(conn: psycopg.Connection):
    """Ensure authentication tables exist and seed initial reviewer accounts."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY,
        email VARCHAR UNIQUE NOT NULL,
        password_hash VARCHAR NOT NULL,
        full_name VARCHAR NOT NULL,
        role VARCHAR NOT NULL DEFAULT 'curator',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

    CREATE TABLE IF NOT EXISTS user_sessions (
        token VARCHAR PRIMARY KEY,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);

    CREATE TABLE IF NOT EXISTS curation_votes (
        id UUID PRIMARY KEY,
        relationship_id UUID NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        vote VARCHAR NOT NULL CHECK (vote IN ('APPROVE', 'REJECT')),
        comment TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_curation_votes_rel_user UNIQUE (relationship_id, user_id)
    );
    CREATE INDEX IF NOT EXISTS idx_curation_votes_relationship ON curation_votes(relationship_id);
    CREATE INDEX IF NOT EXISTS idx_curation_votes_user ON curation_votes(user_id);
    """
    with conn.cursor() as cur:
        cur.execute(schema_sql)
        conn.commit()

        # Seed default reviewer account if missing
        cur.execute("SELECT id FROM users WHERE email = %s", ("reviewer@ebv-kg.org",))
        if not cur.fetchone():
            default_reviewer_id = uuid.uuid4()
            reviewer_pass_hash = hash_password("reviewer123")
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, role) VALUES (%s, %s, %s, %s, %s)",
                (default_reviewer_id, "reviewer@ebv-kg.org", reviewer_pass_hash, "Dr. Harsha (Lead Reviewer)", "curator")
            )
            logger.info("Seeded default reviewer account: reviewer@ebv-kg.org")

        # Seed default admin account if missing
        cur.execute("SELECT id FROM users WHERE email = %s", ("admin@ebv-kg.org",))
        if not cur.fetchone():
            default_admin_id = uuid.uuid4()
            admin_pass_hash = hash_password("admin123")
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, role) VALUES (%s, %s, %s, %s, %s)",
                (default_admin_id, "admin@ebv-kg.org", admin_pass_hash, "EBV System Admin", "admin")
            )
            logger.info("Seeded default admin account: admin@ebv-kg.org")

        conn.commit()

def get_current_user_from_header(authorization: Optional[str] = Header(None), x_session_token: Optional[str] = Header(None), conn: psycopg.Connection = Depends(get_pg_conn)) -> Optional[Dict[str, Any]]:
    """Dependency to retrieve current logged in user from session token."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
    elif x_session_token:
        token = x_session_token.strip()

    if not token:
        return None

    try:
        ensure_tables_and_seed_users(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT u.id, u.email, u.full_name, u.role, s.expires_at
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = %s
                """,
                (token,)
            )
            row = cur.fetchone()
            if not row:
                return None
            if row["expires_at"] < datetime.now(timezone.utc):
                cur.execute("DELETE FROM user_sessions WHERE token = %s", (token,))
                conn.commit()
                return None

            return {
                "id": str(row["id"]),
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "token": token
            }
    except Exception as e:
        logger.warning("Error looking up session user: %s", e)
        return None

# --- Schemas ---

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: Optional[str] = "curator"

class LoginRequest(BaseModel):
    email: str
    password: str

# --- Endpoints ---

@router.post("/auth/signup")
@router.post("/api/auth/signup")
async def signup(req: SignupRequest, conn: psycopg.Connection = Depends(get_pg_conn)):
    """Register a new reviewer / curator account."""
    ensure_tables_and_seed_users(conn)
    email_clean = req.email.strip().lower()
    if not email_clean or "@" not in email_clean:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    try:
        user_id = uuid.uuid4()
        pass_hash = hash_password(req.password)
        role = req.role if req.role in ("curator", "admin") else "curator"

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email_clean,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="An account with this email already exists.")

            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, role) VALUES (%s, %s, %s, %s, %s)",
                (user_id, email_clean, pass_hash, req.full_name.strip(), role)
            )
            conn.commit()

            # Automatically create a session token for immediate login
            token = secrets.token_hex(32)
            expires_at = datetime.now(timezone.utc) + timedelta(days=30)
            cur.execute(
                "INSERT INTO user_sessions (token, user_id, expires_at) VALUES (%s, %s, %s)",
                (token, user_id, expires_at)
            )
            conn.commit()

        return {
            "status": "success",
            "message": "Account created successfully.",
            "token": token,
            "user": {
                "id": str(user_id),
                "email": email_clean,
                "full_name": req.full_name.strip(),
                "role": role
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Signup failed: %s", e)
        raise HTTPException(status_code=500, detail=f"User registration failed: {str(e)}")

@router.post("/auth/login")
@router.post("/api/auth/login")
async def login(req: LoginRequest, conn: psycopg.Connection = Depends(get_pg_conn)):
    """Authenticate user and return a session token."""
    ensure_tables_and_seed_users(conn)
    email_clean = req.email.strip().lower()

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, email, password_hash, full_name, role FROM users WHERE email = %s",
                (email_clean,)
            )
            user = cur.fetchone()
            if not user or not verify_password(req.password, user["password_hash"]):
                raise HTTPException(status_code=401, detail="Invalid email or password.")

            token = secrets.token_hex(32)
            expires_at = datetime.now(timezone.utc) + timedelta(days=30)
            cur.execute(
                "INSERT INTO user_sessions (token, user_id, expires_at) VALUES (%s, %s, %s)",
                (token, user["id"], expires_at)
            )
            conn.commit()

            return {
                "status": "success",
                "token": token,
                "user": {
                    "id": str(user["id"]),
                    "email": user["email"],
                    "full_name": user["full_name"],
                    "role": user["role"]
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@router.get("/auth/me")
@router.get("/api/auth/me")
async def get_me(user: Optional[Dict[str, Any]] = Depends(get_current_user_from_header)):
    """Get current authenticated user info."""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"user": user}

@router.post("/auth/logout")
@router.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None), x_session_token: Optional[str] = Header(None), conn: psycopg.Connection = Depends(get_pg_conn)):
    """Logout current user by revoking session token."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
    elif x_session_token:
        token = x_session_token.strip()

    if token:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_sessions WHERE token = %s", (token,))
                conn.commit()
        except Exception:
            pass
    return {"status": "success", "message": "Logged out successfully."}
