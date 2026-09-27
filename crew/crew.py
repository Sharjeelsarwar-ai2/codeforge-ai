# crew/crew.py
from crewai import Crew, Process
from crew.agents import create_agents
from crew.tasks import create_planning_tasks, create_development_tasks
from tools.file_tools import create_project_folder
from pathlib import Path


class CodeForgeCrew:
    """
    The CodeForge AI Software House Crew.
    Orchestrates all agents through planning and development phases.
    """

    def __init__(self):
        self.agents = create_agents()
        self.project_path = None
        self.planning_results = None
        self.development_results = None

    def setup_project(self, project_name: str) -> str:
        """Create the project workspace folder."""
        result = create_project_folder.run(project_name=project_name)
        self.project_path = result
        return result

    def run_planning_phase(self, inputs: dict) -> dict:
        """
        Run the planning phase: requirement analysis, PRD, design system,
        animation plan, and architecture.
        Returns the results of each planning task.
        """
        planning_tasks = create_planning_tasks(self.agents, inputs)

        planning_crew = Crew(
            agents=[
                self.agents["requirement"],
                self.agents["product_manager"],
                self.agents["designer"],
                self.agents["animation"],
                self.agents["architect"],
            ],
            tasks=planning_tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = planning_crew.kickoff()

        # Store planning tasks for development phase context
        self.planning_tasks = planning_tasks

        # Extract individual task outputs
        self.planning_results = {
            "requirements": planning_tasks[0].output.raw if planning_tasks[0].output else "",
            "prd": planning_tasks[1].output.raw if planning_tasks[1].output else "",
            "design_system": planning_tasks[2].output.raw if planning_tasks[2].output else "",
            "animation_plan": planning_tasks[3].output.raw if planning_tasks[3].output else "",
            "architecture": planning_tasks[4].output.raw if planning_tasks[4].output else "",
            "full_output": result.raw if hasattr(result, "raw") else str(result),
        }

        return self.planning_results

    def run_development_phase(self, inputs: dict) -> dict:
        """
        Run the development phase: code generation, review, QA, debug, documentation.
        Must be called after run_planning_phase.
        """
        if not self.planning_tasks:
            raise ValueError("Planning phase must be run first")

        # Update inputs with project path
        dev_inputs = {**inputs, "project_path": self.project_path}

        development_tasks = create_development_tasks(
            self.agents, dev_inputs, self.planning_tasks
        )

        development_crew = Crew(
            agents=[
                self.agents["developer"],
                self.agents["reviewer"],
                self.agents["qa"],
                self.agents["debug"],
                self.agents["documentation"],
            ],
            tasks=development_tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = development_crew.kickoff()

        self.development_results = {
            "development": development_tasks[0].output.raw if development_tasks[0].output else "",
            "code_review": development_tasks[1].output.raw if development_tasks[1].output else "",
            "qa_report": development_tasks[2].output.raw if development_tasks[2].output else "",
            "debug_fixes": development_tasks[3].output.raw if development_tasks[3].output else "",
            "documentation": development_tasks[4].output.raw if development_tasks[4].output else "",
            "full_output": result.raw if hasattr(result, "raw") else str(result),
        }

        return self.development_results

    def run_full_pipeline(self, inputs: dict, on_phase_complete=None) -> dict:
        """
        Run the complete pipeline: planning → development.
        Optional callback for phase completion notifications.
        """
        # Setup project
        project_name = inputs.get("project_name", "my-project")
        self.setup_project(project_name)
        inputs["project_path"] = self.project_path
        inputs["project_name"] = project_name

        # Phase 1: Planning
        planning_results = self.run_planning_phase(inputs)
        if on_phase_complete:
            on_phase_complete("planning", planning_results)

        # Phase 2: Development
        dev_results = self.run_development_phase(inputs)
        if on_phase_complete:
            on_phase_complete("development", dev_results)

        return {
            "project_path": self.project_path,
            "planning": planning_results,
            "development": dev_results,
        }