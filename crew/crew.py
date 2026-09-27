# crew/crew.py

from crewai import Crew, Process

from crew.agents import create_agents
from crew.tasks import (
    create_planning_tasks,
    create_development_tasks,
)

from tools.file_tools import create_project_folder


class CodeForgeCrew:
    """
    The CodeForge AI Software House Crew.

    Pipeline:

        Planning
        --------
        Requirement
        → Product
        → Design
        → Animation
        → Architecture

        Development
        -----------
        Developer
        → Reviewer
        → QA
        → Debug
        → Documentation
    """

    def __init__(self):
        # Agents are created once for this pipeline.
        self.agents = create_agents()

        self.project_path = None

        self.planning_tasks = None
        self.planning_results = None

        self.development_results = None

    # ----------------------------------------------------------------
    # PROJECT SETUP
    # ----------------------------------------------------------------

    def setup_project(self, project_name: str) -> str:
        """
        Create the project workspace folder.
        """

        result = create_project_folder.run(
            project_name=project_name
        )

        self.project_path = result

        return result

    # ----------------------------------------------------------------
    # PLANNING PHASE
    # ----------------------------------------------------------------

    def run_planning_phase(self, inputs: dict) -> dict:
        """
        Run the five-agent planning phase.

        The tasks run sequentially so each planning stage can use
        the relevant output from the previous stage.
        """

        planning_tasks = create_planning_tasks(
            self.agents,
            inputs,
        )

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

            verbose=False,
        )

        result = planning_crew.kickoff()

        # Store tasks so the development phase can access
        # their outputs.
        self.planning_tasks = planning_tasks

        # ------------------------------------------------------------
        # Extract individual outputs
        # ------------------------------------------------------------

        self.planning_results = {
            "requirements": (
                planning_tasks[0].output.raw
                if planning_tasks[0].output
                else ""
            ),

            "prd": (
                planning_tasks[1].output.raw
                if planning_tasks[1].output
                else ""
            ),

            "design_system": (
                planning_tasks[2].output.raw
                if planning_tasks[2].output
                else ""
            ),

            "animation_plan": (
                planning_tasks[3].output.raw
                if planning_tasks[3].output
                else ""
            ),

            "architecture": (
                planning_tasks[4].output.raw
                if planning_tasks[4].output
                else ""
            ),

            "full_output": (
                result.raw
                if hasattr(result, "raw")
                else str(result)
            ),
        }

        return self.planning_results

    # ----------------------------------------------------------------
    # DEVELOPMENT PHASE
    # ----------------------------------------------------------------

    def run_development_phase(self, inputs: dict) -> dict:
        """
        Run the development phase.

        Planning must have completed successfully first.
        """

        if not self.planning_tasks:
            raise ValueError(
                "Planning phase must be run first"
            )

        # Add project path to development inputs.
        dev_inputs = {
            **inputs,
            "project_path": self.project_path,
        }

        development_tasks = create_development_tasks(
            self.agents,
            dev_inputs,
            self.planning_tasks,
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

            verbose=False,
        )

        result = development_crew.kickoff()

        # ------------------------------------------------------------
        # Extract outputs
        # ------------------------------------------------------------

        self.development_results = {
            "development": (
                development_tasks[0].output.raw
                if development_tasks[0].output
                else ""
            ),

            "code_review": (
                development_tasks[1].output.raw
                if development_tasks[1].output
                else ""
            ),

            "qa_report": (
                development_tasks[2].output.raw
                if development_tasks[2].output
                else ""
            ),

            "debug_fixes": (
                development_tasks[3].output.raw
                if development_tasks[3].output
                else ""
            ),

            "documentation": (
                development_tasks[4].output.raw
                if development_tasks[4].output
                else ""
            ),

            "full_output": (
                result.raw
                if hasattr(result, "raw")
                else str(result)
            ),
        }

        return self.development_results

    # ----------------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------------

    def run_full_pipeline(
        self,
        inputs: dict,
        on_phase_complete=None,
    ) -> dict:
        """
        Run the complete CodeForge pipeline:

            1. Create project
            2. Planning
            3. Development
        """

        # ------------------------------------------------------------
        # Project setup
        # ------------------------------------------------------------

        project_name = inputs.get(
            "project_name",
            "my-project",
        )

        self.setup_project(project_name)

        inputs["project_path"] = self.project_path
        inputs["project_name"] = project_name

        # ------------------------------------------------------------
        # Planning
        # ------------------------------------------------------------

        planning_results = self.run_planning_phase(
            inputs
        )

        if on_phase_complete:
            on_phase_complete(
                "planning",
                planning_results,
            )

        # ------------------------------------------------------------
        # Development
        # ------------------------------------------------------------

        development_results = self.run_development_phase(
            inputs
        )

        if on_phase_complete:
            on_phase_complete(
                "development",
                development_results,
            )

        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------

        return {
            "project_path": self.project_path,
            "planning": planning_results,
            "development": development_results,
        }
