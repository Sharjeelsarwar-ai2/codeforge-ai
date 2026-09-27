# crew/agents.py
"""
Agent creation for the CodeForge AI Software House.

All agents run on Groq's GPT-OSS 120B model.
Behavior is differentiated by temperature and prompt (backstory/goal).
"""

import yaml
from pathlib import Path
from crewai import Agent
from utils.llm import get_llm
from tools.file_tools import (
    create_project_folder,
    write_file,
    read_file,
    update_file,
    create_file_tree,
    list_all_files,
)
from tools.code_tools import (
    validate_html,
    validate_css,
    validate_javascript,
    extract_code_blocks,
    code_formatter,
)
from tools.project_tools import (
    generate_readme,
    generate_gitignore,
    zip_project,
    save_project_metadata,
)
from tools.preview_tools import build_preview_html


def load_agent_configs() -> dict:
    """Load agent configurations from YAML file."""
    config_path = Path(__file__).parent / "config" / "agents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_agents() -> dict:
    """
    Create all AI agents for the software house crew.

    All agents use GPT-OSS 120B via Groq. Personality is differentiated by:
      - temperature (0.2 = strict, 0.5 = creative)
      - role, goal, and backstory (loaded from agents.yaml)
    """
    configs = load_agent_configs()

    # --- LLM tiers (same model, different temperatures) ---
    strict_llm = get_llm(temperature=0.2)     # QA, review, docs, debug
    balanced_llm = get_llm(temperature=0.3)   # coding, planning, architecture
    creative_llm = get_llm(temperature=0.5)   # design, animation

    # --- Tool bundles ---
    file_tools = [
        write_file, read_file, update_file,
        create_file_tree, list_all_files,
    ]
    code_tools = [
        validate_html, validate_css, validate_javascript,
        extract_code_blocks, code_formatter,
    ]
    project_tools_list = [
        generate_readme, generate_gitignore,
        zip_project, save_project_metadata,
    ]

    agents = {}

    # ------------------------------------------------------------------ #
    #  PLANNING AGENTS
    # ------------------------------------------------------------------ #

    # 1. Business Analyst
    cfg = configs["client_requirement_agent"]
    agents["requirement"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=balanced_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # 2. Product Manager
    cfg = configs["product_manager_agent"]
    agents["product_manager"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=balanced_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ------------------------------------------------------------------ #
    #  DESIGN AGENTS  (higher temperature — creativity matters)
    # ------------------------------------------------------------------ #

    # 3. UI/UX Designer
    cfg = configs["ui_ux_designer_agent"]
    agents["designer"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=creative_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # 4. Motion / Animation Designer
    cfg = configs["animation_agent"]
    agents["animation"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=creative_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # 5. Frontend Architect
    cfg = configs["frontend_architect_agent"]
    agents["architect"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=balanced_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ------------------------------------------------------------------ #
    #  DEVELOPMENT AGENTS
    # ------------------------------------------------------------------ #

    # 6. Frontend Developer  ← most important agent
    cfg = configs["frontend_developer_agent"]
    agents["developer"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=balanced_llm,
        verbose=True,
        allow_delegation=False,
        tools=file_tools + [extract_code_blocks],
        max_iter=15,
    )

    # ------------------------------------------------------------------ #
    #  QUALITY AGENTS  (low temperature — strict/consistent)
    # ------------------------------------------------------------------ #

    # 7. Code Reviewer
    cfg = configs["code_reviewer_agent"]
    agents["reviewer"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=strict_llm,
        verbose=True,
        allow_delegation=False,
        tools=[read_file, list_all_files] + code_tools,
        max_iter=5,
    )

    # 8. QA Engineer
    cfg = configs["qa_testing_agent"]
    agents["qa"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=strict_llm,
        verbose=True,
        allow_delegation=False,
        tools=[read_file, list_all_files],
        max_iter=5,
    )

    # 9. Debug / Self-Healing Engineer
    cfg = configs["debug_agent"]
    agents["debug"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=strict_llm,
        verbose=True,
        allow_delegation=False,
        tools=file_tools + code_tools,
        max_iter=10,
    )

    # 10. Documentation Writer
    cfg = configs["documentation_agent"]
    agents["documentation"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=strict_llm,
        verbose=True,
        allow_delegation=False,
        tools=project_tools_list + [create_file_tree],
        max_iter=5,
    )

    return agents