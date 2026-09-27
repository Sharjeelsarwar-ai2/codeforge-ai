# utils/db.py
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("workspace") / "codeforge.db"


def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            project_type TEXT,
            tech_preference TEXT,
            animation_level TEXT,
            project_path TEXT,
            status TEXT DEFAULT 'planning',
            planning_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_project(
    name: str,
    description: str,
    project_type: str,
    tech_preference: str,
    animation_level: str,
    project_path: str,
    status: str = "planning",
    planning_data: dict = None,
) -> int:
    """Save a project to the database. Returns the project ID."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO projects (name, description, project_type, tech_preference,
                            animation_level, project_path, status, planning_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            description,
            project_type,
            tech_preference,
            animation_level,
            project_path,
            status,
            json.dumps(planning_data) if planning_data else None,
        ),
    )

    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return project_id


def update_project_status(project_id: int, status: str, planning_data: dict = None):
    """Update a project's status and optionally its planning data."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    if planning_data:
        cursor.execute(
            """
            UPDATE projects
            SET status = ?, planning_data = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, json.dumps(planning_data), datetime.now().isoformat(), project_id),
        )
    else:
        cursor.execute(
            """
            UPDATE projects
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, datetime.now().isoformat(), project_id),
        )

    conn.commit()
    conn.close()


def get_all_projects() -> list:
    """Get all projects from the database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
    projects = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return projects


def get_project(project_id: int) -> dict:
    """Get a single project by ID."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()

    conn.close()
    return dict(row) if row else None


def delete_project(project_id: int):
    """Delete a project from the database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()