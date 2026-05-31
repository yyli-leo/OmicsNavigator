"""
Base agent class for OmicsNavigator multi-agent system.

Provides common functionality for all specialized agents including:
- LLM integration via LangChain
- Tool management
- Error handling and retry logic
- Execution logging
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class AgentExecutionError(Exception):
    """Exception raised when agent execution fails."""
    pass


class BaseAgent(ABC):
    """
    Base class for all specialized agents in the OmicsNavigator system.

    Each agent should:
    1. Define its system prompt for role-specific behavior
    2. Implement the execute method with its specific logic
    3. Use tools appropriately for its domain
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str,
        temperature: float = 0.7
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent identifier (e.g., "DataAnalyst", "Planner")
            model: LangChain chat model instance
            tools: List of LangChain tools available to this agent
            system_prompt: System prompt defining agent behavior
            temperature: Sampling temperature for LLM (0.0-1.0)
        """
        self.name = name
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.temperature = temperature

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.

        This method should be overridden by subclasses to provide
        agent-specific system prompts.
        """
        pass

    @abstractmethod
    async def execute(self, state: "AgentState") -> "AgentState":
        """
        Execute the agent's primary task.

        This is the main entry point for agent execution. Each agent
        should implement its specific logic here.

        Args:
            state: Current workflow state containing hypothesis, data paths,
                   and outputs from previous agents

        Returns:
            Updated state with agent's output added to appropriate field

        Raises:
            AgentExecutionError: If execution fails
        """
        pass

    def _format_prompt(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> List[BaseMessage]:
        """
        Format messages for the LLM.

        Args:
            user_input: The user's input/question
            context: Optional context dictionary for prompt enhancement

        Returns:
            List of messages formatted for LangChain
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input)
        ]
        return messages

    async def _invoke_llm(self, messages: List[BaseMessage]) -> str:
        """
        Invoke the LLM with error handling.

        Args:
            messages: List of messages to send to LLM

        Returns:
            LLM response content

        Raises:
            AgentExecutionError: If LLM invocation fails
        """
        try:
            response = await self.model.ainvoke(messages)
            return self._extract_text(response.content)
        except Exception as e:
            raise AgentExecutionError(
                f"LLM invocation failed for {self.name}: {str(e)}"
            )

    def _log_execution(self, state: "AgentState", action: str, result: Any) -> None:
        """
        Log execution to state's execution log.

        Args:
            state: The current agent state (TypedDict)
            action: Description of the action performed
            result: Result of the action (will be truncated for memory)
        """
        if state.get("execution_log") is None:
            state["execution_log"] = []

        log_entry = {
            "agent": self.name,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "result_summary": str(result)[:500]  # Truncate for memory
        }
        state["execution_log"].append(log_entry)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.

        Args:
            name: Tool identifier

        Returns:
            Tool instance if found, None otherwise
        """
        return self.tools.get(name)

    @staticmethod
    def _extract_text(content) -> str:
        """
        Normalize LLM response.content to a string.

        Newer Gemini models (e.g. gemini-3.x) may return content as a list
        of parts instead of a plain string. This helper handles both formats.

        Args:
            content: response.content from LLM (str, list, or other)

        Returns:
            Plain string representation of the content
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            return "".join(parts)
        return str(content)

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all tools available to this agent.

        Returns:
            List of all tool instances
        """
        return list(self.tools.values())
