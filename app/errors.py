"""Domain-specific exceptions, translated to HTTP errors in main.py."""


class RAGError(Exception):
    """Base class for pipeline errors."""


class ConfigError(RAGError):
    """Missing or invalid configuration (e.g. no API key)."""


class DocumentLoadError(RAGError):
    """A document could not be parsed or contained no text."""


class UnsupportedFileType(RAGError):
    """File extension is not supported."""


class LLMError(RAGError):
    """The upstream LLM / embedding provider call failed."""
