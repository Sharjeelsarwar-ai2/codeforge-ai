# tools/code_tools.py
import re
import json
from crewai.tools import tool


@tool("validate_html")
def validate_html(html_code: str) -> str:
    """
    Validate HTML code for common issues like unclosed tags, missing DOCTYPE, etc.
    Args:
        html_code: The HTML source code to validate.
    Returns:
        JSON string with validation results and any issues found.
    """
    issues = []

    # Check for DOCTYPE
    if "<!DOCTYPE" not in html_code.upper() and "<!doctype" not in html_code:
        issues.append({"type": "warning", "message": "Missing <!DOCTYPE html> declaration"})

    # Check for essential tags
    essential_tags = ["<html", "<head", "<body", "</html>", "</head>", "</body>"]
    for tag in essential_tags:
        if tag.lower() not in html_code.lower():
            issues.append({"type": "error", "message": f"Missing essential tag: {tag}"})

    # Check for meta charset
    if 'charset' not in html_code.lower():
        issues.append({"type": "warning", "message": "Missing charset meta tag"})

    # Check for viewport meta
    if 'viewport' not in html_code.lower():
        issues.append({"type": "warning", "message": "Missing viewport meta tag for responsiveness"})

    # Check for title tag
    if '<title' not in html_code.lower():
        issues.append({"type": "warning", "message": "Missing <title> tag"})

    # Check for unclosed common tags
    self_closing = {"img", "br", "hr", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
    tag_pattern = re.compile(r'<(/?)(\w+)[^>]*?(/?)>')
    stack = []
    for match in tag_pattern.finditer(html_code):
        is_closing = match.group(1) == "/"
        tag_name = match.group(2).lower()
        is_self_closing = match.group(3) == "/"

        if tag_name in self_closing or is_self_closing:
            continue

        if is_closing:
            if stack and stack[-1] == tag_name:
                stack.pop()
            elif tag_name in stack:
                while stack and stack[-1] != tag_name:
                    unclosed = stack.pop()
                    issues.append({"type": "error", "message": f"Unclosed tag: <{unclosed}>"})
                if stack:
                    stack.pop()
        else:
            stack.append(tag_name)

    for unclosed in stack:
        if unclosed not in {"html", "head", "body"}:
            issues.append({"type": "error", "message": f"Potentially unclosed tag: <{unclosed}>"})

    result = {
        "valid": len([i for i in issues if i["type"] == "error"]) == 0,
        "error_count": len([i for i in issues if i["type"] == "error"]),
        "warning_count": len([i for i in issues if i["type"] == "warning"]),
        "issues": issues,
    }
    return json.dumps(result, indent=2)


@tool("validate_css")
def validate_css(css_code: str) -> str:
    """
    Validate CSS code for common syntax issues.
    Args:
        css_code: The CSS source code to validate.
    Returns:
        JSON string with validation results and any issues found.
    """
    issues = []

    # Check balanced braces
    open_braces = css_code.count("{")
    close_braces = css_code.count("}")
    if open_braces != close_braces:
        issues.append({
            "type": "error",
            "message": f"Unbalanced braces: {open_braces} opening vs {close_braces} closing",
        })

    # Check for missing semicolons (simplified)
    lines = css_code.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("/*")
            and not stripped.startswith("*")
            and not stripped.startswith("//")
            and not stripped.endswith("{")
            and not stripped.endswith("}")
            and not stripped.endswith(",")
            and not stripped.endswith(";")
            and not stripped.startswith("@")
            and ":" in stripped
            and "{" not in stripped
            and "}" not in stripped
        ):
            issues.append({
                "type": "warning",
                "message": f"Line {i + 1}: Possible missing semicolon: '{stripped[:50]}...'",
            })

    # Check for empty rules
    empty_rule = re.findall(r'[^}]+\{\s*\}', css_code)
    for rule in empty_rule:
        selector = rule.split("{")[0].strip()
        issues.append({
            "type": "warning",
            "message": f"Empty CSS rule for selector: '{selector}'",
        })

    result = {
        "valid": len([i for i in issues if i["type"] == "error"]) == 0,
        "error_count": len([i for i in issues if i["type"] == "error"]),
        "warning_count": len([i for i in issues if i["type"] == "warning"]),
        "issues": issues,
    }
    return json.dumps(result, indent=2)


@tool("validate_javascript")
def validate_javascript(js_code: str) -> str:
    """
    Validate JavaScript code for common syntax issues.
    Args:
        js_code: The JavaScript source code to validate.
    Returns:
        JSON string with validation results and any issues found.
    """
    issues = []

    # Check balanced brackets
    brackets = {"(": ")", "[": "]", "{": "}"}
    stack = []
    in_string = False
    string_char = None
    in_comment = False
    in_block_comment = False

    i = 0
    while i < len(js_code):
        char = js_code[i]
        next_char = js_code[i + 1] if i + 1 < len(js_code) else ""

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 1
            i += 1
            continue

        if in_comment:
            if char == "\n":
                in_comment = False
            i += 1
            continue

        if in_string:
            if char == "\\":
                i += 2
                continue
            if char == string_char:
                in_string = False
            i += 1
            continue

        if char == "/" and next_char == "/":
            in_comment = True
            i += 2
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            i += 2
            continue

        if char in ('"', "'", "`"):
            in_string = True
            string_char = char
            i += 1
            continue

        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if stack:
                expected_open = {v: k for k, v in brackets.items()}[char]
                if stack[-1] == expected_open:
                    stack.pop()
                else:
                    issues.append({
                        "type": "error",
                        "message": f"Mismatched bracket: expected closing for '{stack[-1]}', got '{char}'",
                    })
            else:
                issues.append({
                    "type": "error",
                    "message": f"Extra closing bracket: '{char}'",
                })

        i += 1

    for unclosed in stack:
        issues.append({
            "type": "error",
            "message": f"Unclosed bracket: '{unclosed}'",
        })

    # Check for console.log (warning for production)
    console_count = len(re.findall(r'console\.\w+\(', js_code))
    if console_count > 0:
        issues.append({
            "type": "warning",
            "message": f"Found {console_count} console statement(s) — consider removing for production",
        })

    result = {
        "valid": len([i for i in issues if i["type"] == "error"]) == 0,
        "error_count": len([i for i in issues if i["type"] == "error"]),
        "warning_count": len([i for i in issues if i["type"] == "warning"]),
        "issues": issues,
    }
    return json.dumps(result, indent=2)


@tool("extract_code_blocks")
def extract_code_blocks(llm_response: str) -> str:
    """
    Extract all code blocks from an LLM response text.
    Parses markdown fenced code blocks (```language ... ```) and returns them.
    Args:
        llm_response: Raw text response from an LLM containing code blocks.
    Returns:
        JSON string with extracted code blocks and their languages.
    """
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.findall(pattern, llm_response, re.DOTALL)

    blocks = []
    for language, code in matches:
        lang = language.strip().lower() if language.strip() else "text"
        # Normalize language names
        lang_map = {
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "htm": "html",
            "yml": "yaml",
        }
        lang = lang_map.get(lang, lang)

        blocks.append({
            "language": lang,
            "code": code.strip(),
            "line_count": len(code.strip().split("\n")),
        })

    return json.dumps(blocks, indent=2)


@tool("code_formatter")
def code_formatter(code: str, language: str) -> str:
    """
    Format code with proper indentation and spacing.
    Args:
        code: The source code to format.
        language: The programming language (html, css, javascript, python).
    Returns:
        The formatted code string.
    """
    lines = code.split("\n")
    formatted_lines = []
    indent_level = 0
    indent_str = "  "  # 2 spaces

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue

        # Decrease indent for closing brackets/tags
        if language in ("html", "xml"):
            if stripped.startswith("</") or stripped.startswith("/>"):
                indent_level = max(0, indent_level - 1)
        elif language in ("css", "javascript", "json"):
            if stripped.startswith("}") or stripped.startswith("]") or stripped.startswith(")"):
                indent_level = max(0, indent_level - 1)
        elif language == "python":
            if stripped.startswith(("return", "break", "continue", "pass", "raise", "elif", "else", "except", "finally")):
                indent_level = max(0, indent_level - 1)

        formatted_lines.append(f"{indent_str * indent_level}{stripped}")

        # Increase indent for opening brackets/tags
        if language in ("html", "xml"):
            if (
                stripped.startswith("<")
                and not stripped.startswith("</")
                and not stripped.startswith("<!")
                and not stripped.endswith("/>")
                and not any(stripped.startswith(f"<{t}") for t in [
                    "img", "br", "hr", "input", "meta", "link", "area",
                    "base", "col", "embed", "source", "track", "wbr"
                ])
                and "</" not in stripped
            ):
                indent_level += 1
        elif language in ("css", "javascript", "json"):
            if stripped.endswith("{") or stripped.endswith("[") or stripped.endswith("("):
                indent_level += 1
        elif language == "python":
            if stripped.endswith(":"):
                indent_level += 1

    return "\n".join(formatted_lines)