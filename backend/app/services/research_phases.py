"""Explicit research agent phases (Day 11–12 orchestration)."""

from enum import Enum


class ResearchPhase(str, Enum):
    PLANNING = "PLANNING"
    TOOLING = "TOOLING"
    INDEXING = "INDEXING"
    RETRIEVING = "RETRIEVING"
    ANSWERING = "ANSWERING"
    DONE = "DONE"
    FAILED = "FAILED"
