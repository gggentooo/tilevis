"""TraceRecorder orchestrating model preparation, hook attachment, and Trace creation.

Single responsibility: orchestrate probe execution and produce a validated Trace object.
Allowed dependencies: torch, tilevis.core, stdlib.
"""
