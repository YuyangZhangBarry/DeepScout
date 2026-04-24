from backend.app.services.research_phases import ResearchPhase


def test_research_phase_values() -> None:
    assert ResearchPhase.PLANNING.value == "PLANNING"
    assert ResearchPhase.DONE.value == "DONE"
