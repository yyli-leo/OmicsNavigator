"""
API Call Logger for LangChain LLM calls.

This module provides comprehensive logging of all LLM API calls, including:
- Complete input messages
- Full output responses
- Token usage
- Timing information
- Metadata

Logs are saved to ./outputs/api_calls/ with timestamps for traceability.
"""

import os
import json
import logging
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger("omicsnav.api_logger")


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable objects."""

    def default(self, obj: Any) -> Any:
        # Handle UUID
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Handle objects with __dict__ attribute
        if hasattr(obj, '__dict__'):
            return str(obj)

        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)

        # Default to string representation
        return str(obj)


class APICallLogger(BaseCallbackHandler):
    """
    Callback handler to log all LLM API calls to disk.

    Captures complete request/response data for debugging and analysis.
    Supports both LLM and ChatModel callbacks.

    Thread-safe: Uses call tracking dict for concurrent API calls.
    """

    def __init__(self, log_dir: str = "./outputs/api_calls", agent_name: str = "default", model_name: str = ""):
        """
        Initialize the API call logger.

        Args:
            log_dir: Directory to save log files (should be session's api_calls directory)
            agent_name: Name of agent for subdirectory organization
            model_name: Actual model name used by this agent
        """
        self.log_dir = Path(log_dir)
        self.agent_name = agent_name
        self.model_name = model_name

        # Create agent-specific subdirectory
        self.agent_dir = self.log_dir / agent_name
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Use agent_dir as session_dir for saving calls
        self.session_dir = self.agent_dir

        # Thread-safe call tracking using lock and counter
        self._call_count_lock = threading.Lock()
        self._call_count = 0
        self._call_tracking = {}  # call_id -> call_data

        # Enable all callback types
        self.raise_error = False

        # Create agent metadata file
        self._create_agent_metadata()

        logger.debug("Logging %s to: %s", agent_name, self.session_dir)

    @property
    def ignore_llm(self) -> bool:
        """Don't ignore LLM callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Don't ignore chat model callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Ignore chain callbacks."""
        return True

    @property
    def ignore_agent(self) -> bool:
        """Ignore agent callbacks."""
        return True

    @property
    def ignore_retriever(self) -> bool:
        """Ignore retriever callbacks."""
        return True

    def _create_agent_metadata(self):
        """Create a metadata file for this agent's logging session."""
        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("HTTP_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("http_proxy")
        )
        metadata = {
            "agent_name": self.agent_name,
            "start_time": datetime.now().isoformat(),
            "environment": {
                "model": self.model_name or os.getenv("DEFAULT_MODEL", "unknown"),
                "proxy": proxy_url or "disabled",
                "data_root": os.getenv("DATA_ROOT", "./data/s255"),
            }
        }

        metadata_path = self.session_dir / "_agent_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _get_run_id(self, **kwargs) -> str:
        """Extract run_id from kwargs, fallback to thread ID."""
        # LangChain passes run_id in kwargs for callbacks
        run_id = kwargs.get("run_id") or kwargs.get("__run_id__")
        if run_id:
            return str(run_id)
        # Fallback to thread ID if no run_id
        return f"thread_{threading.get_ident()}"

    def on_chat_model_start(self, serialized, messages, **kwargs):
        run_id = self._get_run_id(**kwargs)
        call_id = f"{run_id}_{uuid.uuid4().hex[:8]}"

        # Increment call count (thread-safe)
        with self._call_count_lock:
            self._call_count += 1
            call_number = self._call_count

        # Initialize call tracking for this call
        call_data = {
            "call_id": call_id,
            "run_id": run_id,
            "call_number": call_number,
            "timestamp": datetime.now().isoformat(),
            "start_time": time.time(),
            "model_info": {
                "name": serialized.get("name", "unknown"),
                "kwargs": serialized.get("kwargs", {}),
            },
            "request": {
                "messages": [],
                "invocation_params": kwargs,
            },
            "response": None,
            "timing": {},
            "metadata": {}
        }

        # Store in tracking dict
        self._call_tracking[call_id] = call_data

        # Serialize input messages
        for msg_list in messages:
            for msg in msg_list:
                call_data["request"]["messages"].append(
                    self._serialize_message(msg)
                )

    # LLM callbacks (used by ChatGoogleGenerativeAI and other models)
    # These are the primary callbacks for logging
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        run_id = self._get_run_id(**kwargs)
        call_id = f"{run_id}_{uuid.uuid4().hex[:8]}"

        with self._call_count_lock:
            self._call_count += 1
            call_number = self._call_count

        call_data = {
            "call_id": call_id,
            "run_id": run_id,
            "call_number": call_number,
            "timestamp": datetime.now().isoformat(),
            "start_time": time.time(),
            "model_info": {
                "name": serialized.get("name", "unknown"),
                "kwargs": serialized.get("kwargs", {}),
            },
            "request": {
                "prompts": prompts,
                "invocation_params": kwargs,
            },
            "response": None,
            "timing": {},
            "metadata": {}
        }

        self._call_tracking[call_id] = call_data

    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends (for non-chat models or as fallback)."""
        run_id = self._get_run_id(**kwargs)

        # Find the corresponding start call
        call_data = None
        for cid, data in self._call_tracking.items():
            if data.get("run_id") == run_id or cid.endswith(run_id):
                call_data = data
                break

        if not call_data:
            logger.warning("No start data found for run_id=%s", run_id)
            return

        start_time = call_data.get("start_time", time.time())
        duration = time.time() - start_time

        # Serialize outputs and extract token usage
        outputs = []
        token_usage = {}
        for generations in response.generations:
            for gen in generations:
                output = {
                    "text": gen.text if hasattr(gen, 'text') else str(gen),
                }
                if hasattr(gen, 'generation_info'):
                    output["generation_info"] = gen.generation_info
                outputs.append(output)

                # Extract token usage from ChatGeneration's message
                if hasattr(gen, 'message'):
                    msg = gen.message
                    if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                        um = msg.usage_metadata
                        token_usage = {
                            "prompt_tokens": um.get("input_tokens", 0),
                            "completion_tokens": um.get("output_tokens", 0),
                            "total_tokens": um.get("total_tokens", 0),
                        }

        # Fallback: try response.llm_output (OpenAI-style)
        if not token_usage and response.llm_output:
            if isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get('token_usage', {})

        call_data["response"] = {
            "outputs": outputs,
            "llm_output": response.llm_output,
            "token_usage": token_usage,
        }

        call_data["timing"] = {
            "duration_seconds": round(duration, 3),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
        }

        call_data["metadata"] = kwargs

        # Save to file
        self._save_call(call_data)

    def on_llm_error(self, error, **kwargs):
        """Called when LLM encounters an error."""
        run_id = self._get_run_id(**kwargs)

        # Find the corresponding start call
        call_data = None
        for cid, data in self._call_tracking.items():
            if data.get("run_id") == run_id or cid.endswith(run_id):
                call_data = data
                break

        if not call_data:
            return

        start_time = call_data.get("start_time", time.time())
        duration = time.time() - start_time

        call_data["response"] = {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        }

        call_data["timing"] = {
            "duration_seconds": round(duration, 3),
            "error_occurred": True,
        }

        self._save_call(call_data)

    def on_chat_model_end(self, response, **kwargs):
        # Find call by run_id in _call_tracking
        run_id = self._get_run_id(**kwargs)

        # Find the corresponding start call
        call_data = None
        for cid, data in self._call_tracking.items():
            if data.get("run_id") == run_id or cid.endswith(run_id):
                call_data = data
                break

        if not call_data:
            logger.warning("No start data found for run_id=%s", run_id)
            return

        # Calculate duration
        start_time = call_data.get("start_time", time.time())
        duration = time.time() - start_time

        # Serialize output messages
        outputs = []
        tool_calls_detected = False

        for generations in response.generations:
            for gen in generations:
                msg = gen.message if hasattr(gen, 'message') else gen
                output = self._serialize_message(msg)

                # Check for tool calls in the message
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls_detected = True
                    # Explicitly serialize tool calls
                    output["tool_calls"] = [
                        {
                            "id": str(tc.get('id', '')),  # Convert UUID to string
                            "name": tc.get('name', ''),
                            "args": tc.get('args', {}),
                        }
                        for tc in msg.tool_calls
                    ]

                # Add generation info if available
                if hasattr(gen, 'generation_info'):
                    output["generation_info"] = gen.generation_info

                # Add finish reason if available
                if hasattr(gen, 'finish_reason'):
                    output["finish_reason"] = gen.finish_reason

                outputs.append(output)

        # Extract token usage
        token_usage = {}
        if response.llm_output:
            if isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get('token_usage', {})
                if not token_usage:
                    # Try other common keys
                    for key in ['usage', 'tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens']:
                        if key in response.llm_output:
                            token_usage = {key: response.llm_output[key]}
                            break

        # Update call data with response
        call_data["response"] = {
            "messages": outputs,
            "llm_output": response.llm_output,
            "token_usage": token_usage,
            "has_tool_calls": tool_calls_detected,
        }

        call_data["timing"] = {
            "duration_seconds": round(duration, 3),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
        }

        call_data["metadata"] = kwargs

        # Save to file
        self._save_call(call_data)

    def on_chat_model_error(self, error, **kwargs):
        # Find call by run_id
        run_id = self._get_run_id(**kwargs)

        # Find the corresponding start call
        call_data = None
        for cid, data in self._call_tracking.items():
            if data.get("run_id") == run_id or cid.endswith(run_id):
                call_data = data
                break

        if not call_data:
            logger.warning("No start data found for error in run_id=%s", run_id)
            return

        # Calculate duration
        start_time = call_data.get("start_time", time.time())
        duration = time.time() - start_time

        call_data["response"] = {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            }
        }

        call_data["timing"] = {
            "duration_seconds": round(duration, 3),
            "error_occurred": True,
        }

        # Save to file
        self._save_call(call_data)

    def _save_call(self, call_data):
        # Save and remove from _call_tracking
        # Filename format: call_001_TIMESTAMP.json
        call_number = call_data.get("call_number", 0)
        filename = f"call_{call_number:03d}_{int(time.time())}.json"
        filepath = self.session_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(call_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

            # Also update index file
            self._update_index(filename, call_data)

            logger.debug("Saved call #%d to %s", call_number, filename)

        except Exception as e:
            logger.error("Error saving call: %s", e)
            import traceback
            traceback.print_exc()

        # Remove from tracking
        call_id = call_data.get("call_id")
        if call_id and call_id in self._call_tracking:
            del self._call_tracking[call_id]

    def _update_index(self, filename, call_data):
        # Append to _index.jsonl
        index_path = self.session_dir / "_index.jsonl"

        # Create summary entry
        summary = {
            "filename": filename,
            "call_number": call_data.get("call_number", 0),
            "timestamp": call_data.get("timestamp", ""),
            "model": call_data.get("model_info", {}).get("name", "unknown"),
            "duration_seconds": call_data.get("timing", {}).get("duration_seconds", 0),
            "token_usage": call_data.get("response", {}).get("token_usage", {}),
        }

        try:
            with open(index_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error("Error updating index: %s", e)

    def _serialize_message(self, msg):
        # Keep existing implementation
        result = {
            "type": type(msg).__name__,
            "content": msg.content if hasattr(msg, 'content') else str(msg),
        }

        # Add additional fields based on message type
        if hasattr(msg, 'name') and msg.name:
            result["name"] = msg.name

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": str(tc.get('id', '')),  # Convert UUID to string
                    "name": tc.get('name', ''),
                    "args": tc.get('args', {}),
                }
                for tc in msg.tool_calls
            ]

        if hasattr(msg, 'additional_kwargs'):
            # Filter out non-serializable items
            additional_kwargs = {}
            for key, value in msg.additional_kwargs.items():
                try:
                    json.dumps(value)  # Test if serializable
                    additional_kwargs[key] = value
                except (TypeError, ValueError):
                    additional_kwargs[key] = str(value)  # Convert to string if not serializable
            result["additional_kwargs"] = additional_kwargs

        # Handle response_metadata which may contain UUIDs
        if hasattr(msg, 'response_metadata'):
            response_metadata = {}
            for key, value in msg.response_metadata.items():
                try:
                    json.dumps(value)
                    response_metadata[key] = value
                except (TypeError, ValueError):
                    response_metadata[key] = str(value)
            result["response_metadata"] = response_metadata

        return result


def create_api_logger(log_dir: str = "./outputs/api_calls", agent_name: str = "default", model_name: str = "") -> APICallLogger:
    """
    Create an API call logger instance.

    Args:
        log_dir: Directory to save log files (should be session's api_calls directory)
        agent_name: Name of agent for subdirectory organization

    Returns:
        APICallLogger instance
    """
    return APICallLogger(log_dir, agent_name)


def get_latest_session_dir(base_dir: str = "./outputs/api_calls") -> Optional[Path]:
    """
    Get the most recent logging session directory.

    Args:
        base_dir: Base directory containing all sessions

    Returns:
        Path to the most recent session directory, or None if no sessions exist
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        return None

    # Get all subdirectories
    session_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not session_dirs:
        return None

    # Sort by modification time (most recent first)
    session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return session_dirs[0]


def load_session_summary(session_dir: str) -> Dict[str, Any]:
    """
    Load summary information from a session directory.

    Args:
        session_dir: Path to the session directory

    Returns:
        Dictionary with session summary information
    """
    session_path = Path(session_dir)

    # Load agent metadata if available
    agent_metadata_path = session_path / "_agent_metadata.json"
    agent_metadata = {}
    if agent_metadata_path.exists():
        with open(agent_metadata_path, 'r') as f:
            agent_metadata = json.load(f)

    # Load index if available
    index_path = session_path / "_index.jsonl"
    calls_summary = []
    if index_path.exists():
        with open(index_path, 'r') as f:
            for line in f:
                if line.strip():
                    calls_summary.append(json.loads(line))

    return {
        "session_dir": str(session_path),
        "agent_metadata": agent_metadata,
        "total_calls": len(calls_summary),
        "calls": calls_summary
    }
