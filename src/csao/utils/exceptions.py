class CSAOError(Exception):
    """Base exception for all CSAO related errors."""
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

class ModelLoadError(CSAOError):
    """Raised when a model fails to load."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, error_code="MODEL_LOAD_ERROR", details=details)

class InferenceError(CSAOError):
    """Raised when model inference fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, error_code="INFERENCE_ERROR", details=details)

class ValidationError(CSAOError):
    """Raised when request validation fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)

class AuthenticationError(CSAOError):
    """Raised when API key validation fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", details=details)
