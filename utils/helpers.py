# utils/helpers.py
import re
import json
from pathlib import Path
from datetime import datetime


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-')


def get_file_language(filename: str) -> str:
    """Get the programming language from a filename for syntax highlighting."""
    ext_map = {
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".js": "javascript",
        ".json": "json",
        ".py": "python",
        ".md": "markdown",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".svg": "xml",
        ".gitignore": "text",
    }
    ext = Path(filename).suffix.lower()
    return ext_map.get(ext, "text")


def get_file_icon(filename: str) -> str:
    """Get an emoji icon for a file based on its extension."""
    ext_map = {
        ".html": "🌐",
        ".htm": "🌐",
        ".css": "🎨",
        ".js": "⚡",
        ".json": "📋",
        ".py": "🐍",
        ".md": "📝",
        ".txt": "📄",
        ".gitignore": "🚫",
        ".zip": "📦",
    }
    ext = Path(filename).suffix.lower()
    if filename.startswith("."):
        return "⚙️"
    return ext_map.get(ext, "📄")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_project_stats(project_path: str) -> dict:
    """Get statistics about a project (file count, total size, etc.)."""
    path = Path(project_path)
    if not path.exists():
        return {"error": "Project not found"}

    stats = {
        "total_files": 0,
        "total_size": 0,
        "file_types": {},
        "files": [],
    }

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.name not in {"__pycache__", ".DS_Store"}:
            ext = file_path.suffix.lower() or "no extension"
            size = file_path.stat().st_size

            stats["total_files"] += 1
            stats["total_size"] += size
            stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
            stats["files"].append({
                "name": file_path.name,
                "relative_path": str(file_path.relative_to(path)),
                "full_path": str(file_path.resolve()),
                "size": size,
                "extension": ext,
            })

    stats["total_size_formatted"] = format_file_size(stats["total_size"])
    return stats