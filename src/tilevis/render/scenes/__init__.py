"""Scene protocol, registry, and implementations for each StepKind.

Single responsibility: visualize specific computation steps inside the stage region.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
Forbidden dependencies: torch, tilevis.probe.
"""
