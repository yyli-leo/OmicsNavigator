"""
Pipeline module for OmicsNavigator CLI.
Implements the 7-action spatial omics analysis pipeline demo.
"""

from .orchestrator import PipelineOrchestrator
from .actions import PipelineActions
from .progress import PipelineProgress

__all__ = ['PipelineOrchestrator', 'PipelineActions', 'PipelineProgress']
