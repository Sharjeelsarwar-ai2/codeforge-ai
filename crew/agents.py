# crew/agents.py
"""
Agent creation for the CodeForge AI Software House.

The agents are divided into two major groups:

PLANNING
---------
1. Business Analyst
2. Product Manager
3. UI/UX Designer
4. Animation Planner
5. Frontend Architect

DEVELOPMENT
-----------
6. Frontend Developer
7. Code Reviewer
8. QA Tester
9. Debug Agent
10. Documentation Agent

Planning agents intentionally use:
- lower token budgets
- max_iter=1
- no unnecessary tools

Development agents receive larger token budgets because
they actually generate and modify project files.
"""

import yaml
from pathlib import Path

from crewai import Agent

from utils.llm import (
    get_planning_llm,
    get_development_llm,
    get_review_llm,
    get_debug_llm,
)

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


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

def load_agent_configs() -> dict:
    """Load agent configurations from YAML file."""

    config_path = Path(__file__).parent / "config" / "agents.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------
# AGENT CREATION
# -------------------------------------------------------------------

def create_agents() -> dict:
    """
    Create all CodeForge AI agents.

    Planning agents are deliberately lightweight to reduce Groq
    token consumption.

    Development agents receive larger token budgets because they
    perform actual code generation, review, testing and debugging.
    """

    configs = load_agent_configs()

    # ---------------------------------------------------------------
    # PLANNING LLMs
    # ---------------------------------------------------------------

    planning_balanced = get_planning_llm(
        temperature=0.3
    )

    planning_creative = get_planning_llm(
        temperature=0.5
    )

    planning_strict = get_planning_llm(
        temperature=0.2
    )

    # ---------------------------------------------------------------
    # DEVELOPMENT LLMs
    # ---------------------------------------------------------------

    development_llm = get_development_llm(
        temperature=0.3
    )

    review_llm = get_review_llm(
        temperature=0.2
    )

    debug_llm = get_debug_llm(
        temperature=0.2
    )

    # ---------------------------------------------------------------
    # TOOL BUNDLES
    # ---------------------------------------------------------------

    file_tools = [
        write_file,
        read_file,
        update_file,
        create_file_tree,
        list_all_files,
    ]

    code_tools = [
        validate_html,
        validate_css,
        validate_javascript,
        extract_code_blocks,
        code_formatter,
    ]

    project_tools_list = [
        generate_readme,
        generate_gitignore,
        zip_project,
        save_project_metadata,
    ]

    agents = {}

    # ===============================================================
    # PLANNING AGENTS
    # ===============================================================

    # ---------------------------------------------------------------
    # 1. BUSINESS ANALYST
    # ---------------------------------------------------------------

    cfg = configs["client_requirement_agent"]

    agents["requirement"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=planning_balanced,
        verbose=False,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 2. PRODUCT MANAGER
    # ---------------------------------------------------------------

    cfg = configs["product_manager_agent"]

    agents["product_manager"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=planning_balanced,
        verbose=False,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 3. UI/UX DESIGNER
    # ---------------------------------------------------------------

    cfg = configs["ui_ux_designer_agent"]

    agents["designer"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=planning_creative,
        verbose=False,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 4. ANIMATION AGENT
    # ---------------------------------------------------------------

    cfg = configs["animation_agent"]

    agents["animation"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=planning_creative,
        verbose=False,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 5. FRONTEND ARCHITECT
    # ---------------------------------------------------------------

    cfg = configs["frontend_architect_agent"]

    agents["architect"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=planning_strict,
        verbose=False,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        cache=False,
    )

    # ===============================================================
    # DEVELOPMENT AGENTS
    # ===============================================================

    # ---------------------------------------------------------------
    # 6. FRONTEND DEVELOPER
    # ---------------------------------------------------------------

    cfg = configs["frontend_developer_agent"]

    agents["developer"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=development_llm,
        verbose=False,
        allow_delegation=False,
        tools=file_tools + [extract_code_blocks],
        max_iter=4,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 7. CODE REVIEWER
    # ---------------------------------------------------------------

    cfg = configs["code_reviewer_agent"]

    agents["reviewer"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=review_llm,
        verbose=False,
        allow_delegation=False,
        tools=[
            read_file,
            list_all_files,
        ] + code_tools,
        max_iter=2,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 8. QA TESTING AGENT
    # ---------------------------------------------------------------

    cfg = configs["qa_testing_agent"]

    agents["qa"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=review_llm,
        verbose=False,
        allow_delegation=False,
        tools=[
            read_file,
            list_all_files,
        ],
        max_iter=2,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 9. DEBUG AGENT
    # ---------------------------------------------------------------

    cfg = configs["debug_agent"]

    agents["debug"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=debug_llm,
        verbose=False,
        allow_delegation=False,
        tools=file_tools + code_tools,
        max_iter=3,
        max_retry_limit=0,
        cache=False,
    )

    # ---------------------------------------------------------------
    # 10. DOCUMENTATION AGENT
    # ---------------------------------------------------------------

    cfg = configs["documentation_agent"]

    agents["documentation"] = Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=review_llm,
        verbose=False,
        allow_delegation=False,
        tools=project_tools_list + [
            create_file_tree
        ],
        max_iter=2,
        max_retry_limit=0,
        cache=False,
    )

    return agents
