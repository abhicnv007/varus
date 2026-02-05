"""
calTAD Measurement Pipeline

Automated measurement of calibrated tip-apex distance (calTAD) from hip X-rays.
Requires both AP and lateral views.
"""

from .pipeline import CalTADPipeline, PipelineResult, ViewResult
from .measurement import CalTADCalculator, CalTADResult

__version__ = "0.1.0"
__all__ = [
    "CalTADPipeline",
    "PipelineResult",
    "ViewResult",
    "CalTADCalculator",
    "CalTADResult",
]
