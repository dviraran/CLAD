"""Simulation runner for interactive case playthroughs."""

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

from ..schemas import (
    ActionType,
    CaseSimulation,
    DecisionPoint,
    PhaseId,
    Requestable,
)
from ..utils import get_logger


@dataclass
class SimulationState:
    """Current state of a simulation playthrough."""

    current_phase: PhaseId
    phase_index: int = 0
    decision_index: int = 0
    revealed_requests: list[str] = field(default_factory=list)
    decisions_made: list[dict[str, Any]] = field(default_factory=list)
    total_score: int = 0
    max_possible_score: int = 0
    completed: bool = False


@dataclass
class DecisionResult:
    """Result of a decision."""

    decision_id: str
    selected_option: str
    score: int
    max_score: int
    matched_defendant: bool
    matched_court: bool
    feedback: str


class SimulationRunner:
    """Runs interactive case simulations."""

    PHASE_ORDER = [
        PhaseId.PRESENTATION,
        PhaseId.WORKUP,
        PhaseId.DECISION,
        PhaseId.PROCEDURE,
        PhaseId.POSTOP,
        PhaseId.FOLLOWUP,
    ]

    def __init__(self, case: CaseSimulation | dict[str, Any]):
        """Initialize the simulation runner."""
        self.logger = get_logger("sim.runner")
        self.console = Console()

        if isinstance(case, dict):
            self.case_data = case
        else:
            self.case_data = case.model_dump()

        self.simulation = self.case_data.get("simulation", {})
        self.state = SimulationState(current_phase=PhaseId.PRESENTATION)

    def run_interactive(self) -> dict[str, Any]:
        """Run an interactive simulation session."""
        self._show_introduction()

        while not self.state.completed:
            self._show_current_state()
            self._handle_user_action()

        return self._get_final_results()

    def run_automated(self, decisions: list[str] | None = None) -> dict[str, Any]:
        """Run simulation with automated or pre-specified decisions."""
        decisions = decisions or []
        decision_index = 0

        while not self.state.completed:
            # Get current decision point
            decision_points = self.simulation.get("decision_points", [])

            if self.state.decision_index >= len(decision_points):
                self.state.completed = True
                break

            dp = decision_points[self.state.decision_index]

            # Make decision
            if decision_index < len(decisions):
                selected = decisions[decision_index]
            else:
                # Default to court-endorsed option
                selected = self._get_court_endorsed_option(dp)

            result = self._evaluate_decision(dp, selected)
            self.state.decisions_made.append(result.__dict__)
            self.state.total_score += result.score
            self.state.max_possible_score += result.max_score
            self.state.decision_index += 1
            decision_index += 1

        return self._get_final_results()

    def _show_introduction(self) -> None:
        """Display case introduction."""
        summary = self.case_data.get("summary", {})

        self.console.print(Panel(
            f"[bold]{self.case_data.get('case_name', 'Case Simulation')}[/bold]\n\n"
            f"{summary.get('brief', 'No summary available.')}\n\n"
            f"[dim]Domain: {self.case_data.get('clinical_domain', 'Unknown')}[/dim]\n"
            f"[dim]Jurisdiction: {self.case_data.get('jurisdiction', 'Unknown')}[/dim]",
            title="Case Introduction",
        ))

        # Show initial state
        initial = self.simulation.get("initial_state", {})

        self.console.print("\n[bold]Initial Presentation:[/bold]")
        self.console.print(f"Chief Complaint: {initial.get('chief_complaint', 'Unknown')}")

        if initial.get("history_of_present_illness"):
            self.console.print(f"\nHPI: {initial['history_of_present_illness']}")

        if initial.get("past_medical_history"):
            self.console.print(f"\nPMH: {', '.join(initial['past_medical_history'])}")

        if initial.get("physical_examination"):
            exam = initial["physical_examination"]
            self.console.print("\n[bold]Physical Examination:[/bold]")
            if exam.get("vital_signs"):
                for key, value in exam["vital_signs"].items():
                    if value:
                        self.console.print(f"  {key}: {value}")
            if exam.get("focused_exam"):
                for system, finding in exam["focused_exam"].items():
                    self.console.print(f"  {system}: {finding}")

        self.console.print("\n" + "=" * 60 + "\n")

    def _show_current_state(self) -> None:
        """Display current simulation state."""
        self.console.print(f"\n[cyan]Current Phase: {self.state.current_phase.value.upper()}[/cyan]")
        self.console.print(f"Score: {self.state.total_score}/{self.state.max_possible_score}")

        # Show available requestables
        requestables = self._get_available_requestables()
        if requestables:
            self.console.print("\n[bold]Available Information Requests:[/bold]")
            for i, req in enumerate(requestables, 1):
                revealed = "✓" if req["request_id"] in self.state.revealed_requests else " "
                self.console.print(f"  [{revealed}] {i}. {req['name']} ({req['type']})")

    def _handle_user_action(self) -> None:
        """Handle user input for the current state."""
        decision_points = self.simulation.get("decision_points", [])

        # Check if we have a decision point for current phase
        current_dp = None
        for i, dp in enumerate(decision_points):
            if i == self.state.decision_index:
                current_dp = dp
                break

        if current_dp:
            self._present_decision_point(current_dp)
        else:
            # No more decision points
            self.state.completed = True
            return

    def _present_decision_point(self, dp: dict[str, Any]) -> None:
        """Present a decision point to the user."""
        self.console.print(Panel(
            f"[bold]{dp.get('prompt', 'What would you do?')}[/bold]\n\n"
            f"[dim]Context: {dp.get('clinical_context', '')}[/dim]",
            title=f"Decision Point {dp.get('decision_id', '')}",
        ))

        options = dp.get("options", [])

        self.console.print("\n[bold]Options:[/bold]")
        for i, opt in enumerate(options, 1):
            self.console.print(f"  {i}. {opt.get('description', '')}")
            if opt.get("clinical_reasoning"):
                self.console.print(f"     [dim]{opt['clinical_reasoning']}[/dim]")

        self.console.print(f"\n  R. Request more information")
        self.console.print(f"  S. Show scoring rubric")

        while True:
            choice = Prompt.ask("\nYour choice", default="1")

            if choice.upper() == "R":
                self._handle_information_request()
                continue

            if choice.upper() == "S":
                self._show_scoring_rubric(dp)
                continue

            try:
                option_index = int(choice) - 1
                if 0 <= option_index < len(options):
                    selected_option = options[option_index]["option_id"]
                    result = self._evaluate_decision(dp, selected_option)
                    self._show_decision_feedback(result, dp)
                    self.state.decisions_made.append(result.__dict__)
                    self.state.total_score += result.score
                    self.state.max_possible_score += result.max_score
                    self.state.decision_index += 1
                    break
                else:
                    self.console.print("[red]Invalid option[/red]")
            except ValueError:
                self.console.print("[red]Please enter a number or R/S[/red]")

    def _handle_information_request(self) -> None:
        """Handle a request for more information."""
        requestables = self._get_available_requestables()

        if not requestables:
            self.console.print("[yellow]No additional information available[/yellow]")
            return

        self.console.print("\n[bold]What would you like to request?[/bold]")
        for i, req in enumerate(requestables, 1):
            revealed = "✓" if req["request_id"] in self.state.revealed_requests else " "
            self.console.print(f"  [{revealed}] {i}. {req['name']} ({req['type']})")

        self.console.print(f"  0. Cancel")

        choice = IntPrompt.ask("\nRequest", default=0)

        if choice == 0:
            return

        if 1 <= choice <= len(requestables):
            req = requestables[choice - 1]
            self._reveal_requestable(req)

    def _reveal_requestable(self, req: dict[str, Any]) -> None:
        """Reveal information from a requestable."""
        self.state.revealed_requests.append(req["request_id"])

        reveal = req.get("reveal", {})

        self.console.print(Panel(
            f"[bold]{req['name']}[/bold]\n\n"
            f"{reveal.get('result_summary', 'No results documented')}\n\n"
            f"[dim]Detailed: {reveal.get('detailed_findings', 'N/A')}[/dim]\n"
            f"[dim]Significance: {reveal.get('clinical_significance', 'N/A')}[/dim]",
            title=f"Results: {req['type']}",
        ))

        if not req.get("was_ordered_in_case", True):
            self.console.print(
                "[yellow]Note: This investigation was NOT ordered in the actual case[/yellow]"
            )

    def _get_available_requestables(self) -> list[dict[str, Any]]:
        """Get requestables available in current phase."""
        all_requestables = self.simulation.get("requestables", [])
        current_phase = self.state.current_phase.value

        available = []
        phase_index = self.PHASE_ORDER.index(self.state.current_phase)

        for req in all_requestables:
            req_phase = req.get("available_phase", "presentation")
            try:
                req_phase_index = self.PHASE_ORDER.index(PhaseId(req_phase))
                if req_phase_index <= phase_index:
                    available.append(req)
            except (ValueError, KeyError):
                available.append(req)

        return available

    def _evaluate_decision(
        self,
        dp: dict[str, Any],
        selected_option: str,
    ) -> DecisionResult:
        """Evaluate a decision and calculate score."""
        options = dp.get("options", [])
        rubric = dp.get("scoring_rubric", {})
        max_score = rubric.get("max_score", 10)

        # Find selected option details
        selected = None
        for opt in options:
            if opt.get("option_id") == selected_option:
                selected = opt
                break

        if not selected:
            return DecisionResult(
                decision_id=dp.get("decision_id", ""),
                selected_option=selected_option,
                score=0,
                max_score=max_score,
                matched_defendant=False,
                matched_court=False,
                feedback="Invalid option selected",
            )

        matched_defendant = selected.get("is_defendant_choice", False)
        matched_court = selected.get("is_court_endorsed", False)

        # Calculate score
        score = 0
        if matched_court:
            score = max_score
        elif matched_defendant:
            score = max_score // 4  # Partial credit for matching actual practice
        else:
            score = max_score // 2  # Some credit for other valid options

        # Generate feedback
        if matched_court:
            feedback = "Excellent! This matches the court-endorsed standard of care."
        elif matched_defendant:
            feedback = "This matches what the defendant did, but fell below the standard of care."
        else:
            feedback = "Consider the available evidence and standard of care guidelines."

        return DecisionResult(
            decision_id=dp.get("decision_id", ""),
            selected_option=selected_option,
            score=score,
            max_score=max_score,
            matched_defendant=matched_defendant,
            matched_court=matched_court,
            feedback=feedback,
        )

    def _show_decision_feedback(
        self,
        result: DecisionResult,
        dp: dict[str, Any],
    ) -> None:
        """Show feedback after a decision."""
        color = "green" if result.matched_court else ("yellow" if result.matched_defendant else "red")

        self.console.print(f"\n[{color}]Score: {result.score}/{result.max_score}[/{color}]")
        self.console.print(f"{result.feedback}\n")

        # Show what actually happened
        actual = dp.get("actual_action_defendant", {})
        expected = dp.get("expected_action_court", {})
        explanation = dp.get("explanation", {})

        self.console.print("[bold]What the defendant actually did:[/bold]")
        self.console.print(f"  {actual.get('description', 'Not documented')}")
        if actual.get("reasoning_stated"):
            self.console.print(f"  [dim]Reasoning: {actual['reasoning_stated']}[/dim]")

        self.console.print("\n[bold]What the court determined should have been done:[/bold]")
        self.console.print(f"  {expected.get('description', 'Not documented')}")
        if expected.get("standard_of_care_basis"):
            self.console.print(f"  [dim]Basis: {expected['standard_of_care_basis']}[/dim]")

        if explanation:
            self.console.print("\n[bold]Explanation:[/bold]")
            if explanation.get("why_defendant_wrong"):
                self.console.print(f"  Why wrong: {explanation['why_defendant_wrong']}")
            if explanation.get("what_should_have_happened"):
                self.console.print(f"  Should have: {explanation['what_should_have_happened']}")

        self.console.print("\n" + "=" * 60 + "\n")

    def _show_scoring_rubric(self, dp: dict[str, Any]) -> None:
        """Display the scoring rubric."""
        rubric = dp.get("scoring_rubric", {})

        table = Table(title="Scoring Rubric")
        table.add_column("Criterion", style="cyan")
        table.add_column("Points", style="green")
        table.add_column("Explanation", style="dim")

        for criterion in rubric.get("criteria", []):
            table.add_row(
                criterion.get("criterion", ""),
                str(criterion.get("points", 0)),
                criterion.get("explanation", ""),
            )

        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{rubric.get('max_score', 0)}[/bold]",
            "",
        )

        self.console.print(table)

    def _get_court_endorsed_option(self, dp: dict[str, Any]) -> str:
        """Get the court-endorsed option ID."""
        for opt in dp.get("options", []):
            if opt.get("is_court_endorsed"):
                return opt.get("option_id", "")
        # Fall back to first option
        options = dp.get("options", [])
        return options[0].get("option_id", "") if options else ""

    def _get_final_results(self) -> dict[str, Any]:
        """Generate final results summary."""
        percentage = (
            self.state.total_score / self.state.max_possible_score * 100
            if self.state.max_possible_score > 0
            else 0
        )

        # Count matches
        court_matches = sum(1 for d in self.state.decisions_made if d.get("matched_court"))
        defendant_matches = sum(1 for d in self.state.decisions_made if d.get("matched_defendant"))

        results = {
            "case_id": self.case_data.get("case_id"),
            "total_score": self.state.total_score,
            "max_score": self.state.max_possible_score,
            "percentage": percentage,
            "decisions_count": len(self.state.decisions_made),
            "court_endorsed_matches": court_matches,
            "defendant_matches": defendant_matches,
            "information_requested": len(self.state.revealed_requests),
            "decisions": self.state.decisions_made,
        }

        self._show_final_summary(results)

        return results

    def _show_final_summary(self, results: dict[str, Any]) -> None:
        """Display final simulation summary."""
        end_state = self.simulation.get("end_state", {})
        patient_outcome = end_state.get("patient_outcome", {})
        legal_outcome = end_state.get("legal_outcome", {})
        malpractice = end_state.get("malpractice_determination", {})

        self.console.print(Panel(
            f"[bold]Final Score: {results['total_score']}/{results['max_score']} "
            f"({results['percentage']:.1f}%)[/bold]\n\n"
            f"Decisions made: {results['decisions_count']}\n"
            f"Court-endorsed matches: {results['court_endorsed_matches']}\n"
            f"Defendant matches: {results['defendant_matches']}\n"
            f"Information requests: {results['information_requested']}",
            title="Simulation Complete",
        ))

        self.console.print("\n[bold]Actual Case Outcome:[/bold]")
        self.console.print(f"Patient: {patient_outcome.get('description', 'Unknown')}")
        self.console.print(f"Verdict: {legal_outcome.get('verdict', 'Unknown')}")

        if malpractice.get("point_of_failure"):
            self.console.print(f"\n[bold]Point of Failure:[/bold]")
            self.console.print(f"  {malpractice['point_of_failure']}")

        if malpractice.get("counterfactual"):
            self.console.print(f"\n[bold]What Should Have Happened:[/bold]")
            self.console.print(f"  {malpractice['counterfactual']}")
