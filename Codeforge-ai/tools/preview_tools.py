# tools/preview_tools.py
import re
from pathlib import Path
from crewai.tools import tool


@tool("build_preview_html")
def build_preview_html(project_path: str) -> str:
    """
    Combine all HTML/CSS/JS files from the project into a single renderable HTML
    string suitable for iframe preview.
    Args:
        project_path: Root path of the project folder.
    Returns:
        A single complete HTML string with all CSS/JS inlined.
    """
    path = Path(project_path)
    if not path.exists():
        return f"<html><body><h1>Error: Project not found at {project_path}</h1></body></html>"

    # Find the main HTML file
    html_file = None
    for candidate in ["index.html", "home.html", "main.html"]:
        if (path / candidate).exists():
            html_file = path / candidate
            break

    if not html_file:
        # Try to find any HTML file
        html_files = list(path.glob("*.html"))
        if html_files:
            html_file = html_files[0]
        else:
            return "<html><body><h1>No HTML file found in project</h1></body></html>"

    html_content = html_file.read_text(encoding="utf-8")

    # Inline CSS files
    css_files = list(path.rglob("*.css"))
    for css_file in css_files:
        css_content = css_file.read_text(encoding="utf-8")
        relative_path = css_file.relative_to(path)

        # Replace link tag with inline style
        link_patterns = [
            f'<link[^>]*href=["\']\.?/?{re.escape(str(relative_path))}["\'][^>]*/?>',
            f'<link[^>]*href=["\']\.?/?{re.escape(css_file.name)}["\'][^>]*/?>',
        ]
        for pattern in link_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                html_content = re.sub(
                    pattern,
                    f"<style>\n{css_content}\n</style>",
                    html_content,
                    flags=re.IGNORECASE,
                )
                break
        else:
            # If no link tag found, inject before </head>
            if css_content.strip():
                html_content = html_content.replace(
                    "</head>",
                    f"<style>\n{css_content}\n</style>\n</head>",
                )

    # Inline JS files
    js_files = list(path.rglob("*.js"))
    for js_file in js_files:
        js_content = js_file.read_text(encoding="utf-8")
        relative_path = js_file.relative_to(path)

        # Replace script tag with inline script
        script_patterns = [
            f'<script[^>]*src=["\']\.?/?{re.escape(str(relative_path))}["\'][^>]*></script>',
            f'<script[^>]*src=["\']\.?/?{re.escape(js_file.name)}["\'][^>]*></script>',
        ]
        for pattern in script_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                html_content = re.sub(
                    pattern,
                    f"<script>\n{js_content}\n</script>",
                    html_content,
                    flags=re.IGNORECASE,
                )
                break
        else:
            # If no script tag found, inject before </body>
            if js_content.strip():
                html_content = html_content.replace(
                    "</body>",
                    f"<script>\n{js_content}\n</script>\n</body>",
                )

    return html_content