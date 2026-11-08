"""Scene registry mapping StepKind to Scene classes with GenericScene fallback.

Single responsibility: register and look up Scene implementations by StepKind.
Allowed dependencies: tilevis.core, stdlib.
"""
