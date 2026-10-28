class StudyError(RuntimeError):
    """Base exception for a study run that cannot be completed faithfully."""


class ConfigurationError(StudyError):
    """Raised when a configuration is missing or internally inconsistent."""


class ModelConsistencyError(StudyError):
    """Raised when literal model constraints are mutually infeasible."""


class OptionalDependencyError(StudyError):
    """Raised when an optional export or optimization dependency is absent."""


class BaselineUnavailableError(StudyError):
    """Raised when a named baseline has no faithful implementation available."""
