# crew/tasks.py
import yaml
from pathlib import Path
from crewai import Task


def load_task_configs() -> dict:
    """Load task configurations from YAML file."""
    config_path = Path(__file__).parent / "config" / "tasks.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_planning_tasks(agents: dict, inputs: dict) -> list:
    """
    Create the planning phase tasks (before coding).
    These run first to generate the project plan for user approval.
    """
    configs = load_task_configs()

    # Task 1: Requirement Analysis
    requirement_task = Task(
        description=configs["requirement_analysis_task"]["description"].format(**inputs),
        expected_output=configs["requirement_analysis_task"]["expected_output"],
        agent=agents["requirement"],
    )

    # Task 2: Product Planning (depends on requirements)
    product_task = Task(
        description=configs["product_planning_task"]["description"],
        expected_output=configs["product_planning_task"]["expected_output"],
        agent=agents["product_manager"],
        context=[requirement_task],
    )

    # Task 3: Design System (depends on requirements + product plan)
    design_task = Task(
        description=configs["design_system_task"]["description"].format(**inputs),
        expected_output=configs["design_system_task"]["expected_output"],
        agent=agents["designer"],
        context=[requirement_task, product_task],
    )

    # Task 4: Animation Planning (depends on design + product plan)
    animation_task = Task(
        description=configs["animation_planning_task"]["description"].format(**inputs),
        expected_output=configs["animation_planning_task"]["expected_output"],
        agent=agents["animation"],
        context=[requirement_task, product_task, design_task],
    )

    # Task 5: Architecture Planning (depends on all above)
    architecture_task = Task(
        description=configs["architecture_planning_task"]["description"],
        expected_output=configs["architecture_planning_task"]["expected_output"],
        agent=agents["architect"],
        context=[requirement_task, product_task, design_task, animation_task],
    )

    return [requirement_task, product_task, design_task, animation_task, architecture_task]


def create_development_tasks(agents: dict, inputs: dict, planning_tasks: list) -> list:
    """
    Create the development phase tasks (after user approves the plan).
    These actually generate, review, test, fix and document the code.
    """
    configs = load_task_configs()

    # Task 6: Frontend Development (depends on all planning tasks)
    development_task = Task(
        description=configs["frontend_development_task"]["description"].format(**inputs),
        expected_output=configs["frontend_development_task"]["expected_output"],
        agent=agents["developer"],
        context=planning_tasks,
    )

    # Task 7: Code Review (depends on development)
    review_task = Task(
        description=configs["code_review_task"]["description"].format(**inputs),
        expected_output=configs["code_review_task"]["expected_output"],
        agent=agents["reviewer"],
        context=[development_task],
    )

    # Task 8: QA Testing (depends on development)
    qa_task = Task(
        description=configs["qa_testing_task"]["description"].format(**inputs),
        expected_output=configs["qa_testing_task"]["expected_output"],
        agent=agents["qa"],
        context=[development_task],
    )

    # Task 9: Debug & Fix (depends on review + QA)
    debug_task = Task(
        description=configs["debug_and_fix_task"]["description"].format(**inputs),
        expected_output=configs["debug_and_fix_task"]["expected_output"],
        agent=agents["debug"],
        context=[development_task, review_task, qa_task],
    )

    # Task 10: Documentation (depends on everything)
    doc_inputs = {
        **inputs,
        "project_name": inputs.get("project_name", "Generated Project"),
        "project_description": inputs.get("user_idea", "An AI-generated web project"),
    }
    documentation_task = Task(
        description=configs["documentation_task"]["description"].format(**doc_inputs),
        expected_output=configs["documentation_task"]["expected_output"],
        agent=agents["documentation"],
        context=[development_task, debug_task],
    )

    return [development_task, review_task, qa_task, debug_task, documentation_task]