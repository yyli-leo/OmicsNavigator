"""
OmicsNavigator Workflows Module

LangGraph-based workflow orchestration for multi-agent pipelines.
"""

# Import AgentState to avoid circular imports
# LangGraphOrchestrator is not imported here to avoid circular dependency
from .agent_state import AgentState

__all__ = ['AgentState']
