# tools/file_tools.py
import os
import json
from pathlib import Path
from crewai.tools import tool

WORKSPACE_DIR = Path("workspace")


@tool("create_project_folder")
def create_project_folder(project_name: str) -> str:
    """
    Create a unique project workspace folder with standard subdirectories.
    Args:
        project_name: Name of the project (will be slugified).
    Returns:
        The absolute path of the created project folder.
    """
    slug = project_name.lower().replace(" ", "-").replace("_", "-")
    # Remove any non-alphanumeric characters except hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    project_path = WORKSPACE_DIR / slug

    counter = 1
    original_path = project_path
    while project_path.exists():
        project_path = Path(f"{original_path}-{counter}")
        counter += 1

    # Create standard subdirectories
    subdirs = ["css", "js", "images", "assets", "components"]
    for subdir in subdirs:
        (project_path / subdir).mkdir(parents=True, exist_ok=True)

    return str(project_path.resolve())


@tool("write_file")
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file at the specified path, creating parent directories if needed.
    Args:
        file_path: Full path to the file to write.
        content: The content to write into the file.
    Returns:
        Confirmation message with the file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {file_path}"


@tool("read_file")
def read_file(file_path: str) -> str:
    """
    Read and return the content of a file.
    Args:
        file_path: Full path to the file to read.
    Returns:
        The file content as a string, or an error message.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found at {file_path}"
    if not path.is_file():
        return f"Error: {file_path} is not a file"
    return path.read_text(encoding="utf-8")


@tool("update_file")
def update_file(file_path: str, old_content: str, new_content: str) -> str:
    """
    Update a specific portion of a file by replacing old_content with new_content.
    Args:
        file_path: Full path to the file to update.
        old_content: The exact text to find and replace.
        new_content: The replacement text.
    Returns:
        Confirmation or error message.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found at {file_path}"

    current = path.read_text(encoding="utf-8")
    if old_content not in current:
        return f"Error: Could not find the specified content to replace in {file_path}"

    updated = current.replace(old_content, new_content, 1)
    path.write_text(updated, encoding="utf-8")
    return f"Successfully updated {file_path}"


@tool("create_file_tree")
def create_file_tree(project_path: str) -> str:
    """
    Generate a visual file tree representation of the project directory.
    Args:
        project_path: Root path of the project folder.
    Returns:
        A formatted string showing the directory tree.
    """
    path = Path(project_path)
    if not path.exists():
        return f"Error: Directory not found at {project_path}"

    lines = []
    _build_tree(path, lines, prefix="")
    return "\n".join(lines)


def _build_tree(directory: Path, lines: list, prefix: str = ""):
    """Recursively build tree lines."""
    entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    entries = [e for e in entries if e.name not in {"__pycache__", ".DS_Store", ".git"}]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _build_tree(entry, lines, prefix + extension)


@tool("list_all_files")
def list_all_files(project_path: str) -> str:
    """
    List all files in the project directory with their relative paths.
    Args:
        project_path: Root path of the project folder.
    Returns:
        JSON string with list of file paths and their sizes.
    """
    path = Path(project_path)
    if not path.exists():
        return f"Error: Directory not found at {project_path}"

    files = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.name not in {"__pycache__", ".DS_Store"}:
            relative = file_path.relative_to(path)
            files.append(
                {
                    "path": str(relative),
                    "full_path": str(file_path.resolve()),
                    "size_bytes": file_path.stat().st_size,
                    "extension": file_path.suffix,
                }
            )

    return json.dumps(files, indent=2)