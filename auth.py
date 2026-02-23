"""
Brain Buddy - Authentication Module
Handles user registration, login, and JWT token management using SQLite.
"""

import sqlite3
import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from jose import JWTError, jwt
from fastapi import HTTPException, status

# ── Config ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "brainbuddy-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

DB_PATH = Path(__file__).parent / "brainbuddy.db"


# ── Database setup ────────────────────────────────────────────────────────────
def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    NOT NULL,
                email       TEXT    NOT NULL UNIQUE,
                password    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ── Password helpers ──────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode a JWT token. Raises HTTPException if invalid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── User CRUD ─────────────────────────────────────────────────────────────────
def create_user(username: str, email: str, password: str) -> dict:
    """Register a new user. Returns the user dict."""
    conn = get_db()
    try:
        # Check if email already exists
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed = hash_password(password)
        now = datetime.utcnow().isoformat()
        cursor = conn.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (username, email.lower(), hashed, now)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {"id": user_id, "username": username, "email": email.lower()}
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> dict:
    """Authenticate a user. Returns user dict or raises HTTPException."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if not verify_password(password, row["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return {"id": row["id"], "username": row["username"], "email": row["email"]}
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict:
    """Get user info by ID."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return dict(row)
    finally:
        conn.close()
