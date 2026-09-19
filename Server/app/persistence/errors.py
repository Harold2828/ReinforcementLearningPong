class EvolutionStoreError(Exception):
    """Base error for the persistence subsystem."""


class ControlledRecoveryError(EvolutionStoreError):
    """Raised when checkpoint integrity cannot be satisfied and a controlled recovery path is required.

    The recorded recovery decision is always applied to the store before this error is raised.
    """

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


class ProvenanceViolationError(EvolutionStoreError):
    """Raised when a record would link entities from different runs or agents.

    Raw match, evaluation, and champion evidence must stay internally consistent.
    """