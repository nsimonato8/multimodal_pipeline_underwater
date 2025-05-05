class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class InputError(PipelineError):
    """Exception raised for errors in the input."""
    pass

class ProcessingError(PipelineError):
    """Exception raised during processing."""
    pass

class RoboflowError(PipelineError):
    """Exception raised during Roboflow API calls."""
    pass