# crew/tasks.py

import yaml
from pathlib import Path

from crewai import Task


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

def load_task_configs() -> dict:
    """Load task configurations from YAML file."""

    config_path = Path(__file__).parent / "config" / "tasks.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------
# PLANNING TASKS
# -------------------------------------------------------------------

def create_planning_tasks(agents: dict, inputs: dict) -> list:
    """
    Create the planning phase tasks.

    Context is intentionally kept narrow.

    Instead of repeatedly sending the complete history to every
    agent, each agent receives only the planning information it
    actually needs.

    Flow:

        Requirements
             ↓
        Product Plan
             ↓
        Design System
             ↓
        Animation Plan
             ↓
        Architecture
    """

    configs = load_task_configs()

    # ---------------------------------------------------------------
    # TASK 1 — REQUIREMENTS
    # ---------------------------------------------------------------

    requirement_task = Task(
        description=configs[
            "requirement_analysis_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "requirement_analysis_task"
        ]["expected_output"],

        agent=agents["requirement"],
    )

    # ---------------------------------------------------------------
    # TASK 2 — PRODUCT PLAN
    # ---------------------------------------------------------------

    product_task = Task(
        description=configs[
            "product_planning_task"
        ]["description"],

        expected_output=configs[
            "product_planning_task"
        ]["expected_output"],

        agent=agents["product_manager"],

        # Only requirements are needed here.
        context=[
            requirement_task
        ],
    )

    # ---------------------------------------------------------------
    # TASK 3 — DESIGN SYSTEM
    # ---------------------------------------------------------------

    design_task = Task(
        description=configs[
            "design_system_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "design_system_task"
        ]["expected_output"],

        agent=agents["designer"],

        # Design mainly depends on the product direction.
        # This avoids sending the raw requirements twice.
        context=[
            product_task
        ],
    )

    # ---------------------------------------------------------------
    # TASK 4 — ANIMATION PLAN
    # ---------------------------------------------------------------

    animation_task = Task(
        description=configs[
            "animation_planning_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "animation_planning_task"
        ]["expected_output"],

        agent=agents["animation"],

        # Animation needs the design system and product direction.
        context=[
            product_task,
            design_task,
        ],
    )

    # ---------------------------------------------------------------
    # TASK 5 — ARCHITECTURE
    # ---------------------------------------------------------------

    architecture_task = Task(
        description=configs[
            "architecture_planning_task"
        ]["description"],

        expected_output=configs[
            "architecture_planning_task"
        ]["expected_output"],

        agent=agents["architect"],

        # Architecture needs the actual design/development direction.
        # Requirements are already represented through the PRD.
        context=[
            product_task,
            design_task,
            animation_task,
        ],
    )

    return [
        requirement_task,
        product_task,
        design_task,
        animation_task,
        architecture_task,
    ]


# -------------------------------------------------------------------
# DEVELOPMENT TASKS
# -------------------------------------------------------------------

def create_development_tasks(
    agents: dict,
    inputs: dict,
    planning_tasks: list,
) -> list:
    """
    Create the development phase tasks.

    Development needs the complete planning output because the
    developer must translate the specification into actual files.

    Review, QA, debugging and documentation then operate on the
    generated project rather than repeatedly receiving every planning
    document.
    """

    configs = load_task_configs()

    # ---------------------------------------------------------------
    # TASK 6 — DEVELOPMENT
    # ---------------------------------------------------------------

    development_task = Task(
        description=configs[
            "frontend_development_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "frontend_development_task"
        ]["expected_output"],

        agent=agents["developer"],

        # The developer needs the complete planning package.
        context=planning_tasks,
    )

    # ---------------------------------------------------------------
    # TASK 7 — CODE REVIEW
    # ---------------------------------------------------------------

    review_task = Task(
        description=configs[
            "code_review_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "code_review_task"
        ]["expected_output"],

        agent=agents["reviewer"],

        # Review the actual development result.
        context=[
            development_task
        ],
    )

    # ---------------------------------------------------------------
    # TASK 8 — QA
    # ---------------------------------------------------------------

    qa_task = Task(
        description=configs[
            "qa_testing_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "qa_testing_task"
        ]["expected_output"],

        agent=agents["qa"],

        # QA only needs the generated project.
        context=[
            development_task
        ],
    )

    # ---------------------------------------------------------------
    # TASK 9 — DEBUG
    # ---------------------------------------------------------------

    debug_task = Task(
        description=configs[
            "debug_and_fix_task"
        ]["description"].format(**inputs),

        expected_output=configs[
            "debug_and_fix_task"
        ]["expected_output"],

        agent=agents["debug"],

        # Debug needs development + review + QA findings.
        context=[
            development_task,
            review_task,
            qa_task,
        ],
    )

    # ---------------------------------------------------------------
    # TASK 10 — DOCUMENTATION
    # ---------------------------------------------------------------

    doc_inputs = {
        **inputs,
        "project_name": inputs.get(
            "project_name",
            "Generated Project",
        ),
        "project_description": inputs.get(
            "user_idea",
            "An AI-generated web project",
        ),
    }

    documentation_task = Task(
        description=configs[
            "documentation_task"
        ]["description"].format(**doc_inputs),

        expected_output=configs[
            "documentation_task"
        ]["expected_output"],

        agent=agents["documentation"],

        # Documentation only needs the development/debug state.
        context=[
            development_task,
            debug_task,
        ],
    )

    return [
        development_task,
        review_task,
        qa_task,
        debug_task,
        documentation_task,
    ]
