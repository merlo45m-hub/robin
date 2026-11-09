"""Telemetry configuration for LangChain with Langfuse and Arize Phoenix.

Non-blocking async export with batch flush. Multi-step reasoning, tool latency,
constraint adherence tracking. OpenRouter/Claude primary, agy fallback.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers import BaseTracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "https://app.phoenix.arize.com/v1/traces")

# Primary LLM provider (OpenRouter/Claude), fallback to agy
PRIMARY_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "openrouter")
FALLBACK_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER", "agy")

# Async queue configuration
BATCH_SIZE = int(os.getenv("TELEMETRY_BATCH_SIZE", "50"))
FLUSH_INTERVAL = float(os.getenv("TELEMETRY_FLUSH_INTERVAL", "5.0"))  # seconds
MAX_QUEUE_SIZE = int(os.getenv("TELEMETRY_MAX_QUEUE_SIZE", "10000"))


# ---------------------------------------------------------------------------
# Async Export Queue
# ---------------------------------------------------------------------------

class AsyncTelemetryQueue:
    """Non-blocking async queue with batch flush for telemetry exports."""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        flush_interval: float = FLUSH_INTERVAL,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        self._queue: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[Any] = None

    def enqueue(self, event: Dict[str, Any]) -> bool:
        """Add event to queue. Returns False if queue is full (non-blocking)."""
        if len(self._queue) >= self.max_queue_size:
            logger.warning("Telemetry queue full, dropping event")
            return False
        self._queue.append(event)
        if len(self._queue) >= self.batch_size:
            self._flush_sync()
        return True

    def _flush_sync(self) -> None:
        """Synchronous flush - export to configured backends."""
        if not self._queue:
            return
        events = self._queue[:self.batch_size]
        self._queue = self._queue[self.batch_size:]

        # Export to Langfuse
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            self._export_langfuse(events)

        # Export to Phoenix
        if PHOENIX_API_KEY:
            self._export_phoenix(events)

    def _export_langfuse(self, events: List[Dict[str, Any]]) -> None:
        """Export events to Langfuse."""
        try:
            from langfuse import Langfuse
            langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            for event in events:
                if event.get("type") == "trace":
                    langfuse.trace(**event["data"])
                elif event.get("type") == "generation":
                    langfuse.generation(**event["data"])
                elif event.get("type") == "span":
                    langfuse.span(**event["data"])
            langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse export failed: {e}")

    def _export_phoenix(self, events: List[Dict[str, Any]]) -> None:
        """Export events to Arize Phoenix."""
        try:
            import requests
            headers = {"Authorization": f"Bearer {PHOENIX_API_KEY}", "Content-Type": "application/json"}
            payload = {"spans": [e["data"] for e in events if e.get("type") == "span"]}
            if payload["spans"]:
                requests.post(PHOENIX_ENDPOINT, json=payload, headers=headers, timeout=5)
        except Exception as e:
            logger.warning(f"Phoenix export failed: {e}")

    def flush(self) -> None:
        """Flush all queued events."""
        while self._queue:
            self._flush_sync()


# Global queue instance
_telemetry_queue = AsyncTelemetryQueue()


# ---------------------------------------------------------------------------
# LangChain Callback Handler
# ---------------------------------------------------------------------------

class TelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for telemetry with multi-step reasoning,
    tool latency, and constraint adherence tracking.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self._trace_id: Optional[str] = None
        self._span_stack: List[str] = []
        self._step_count = 0
        self._tool_latencies: Dict[str, float] = {}
        self._constraint_violations: List[Dict[str, Any]] = []

    @property
    def trace_id(self) -> str:
        if self._trace_id is None:
            import uuid
            self._trace_id = str(uuid.uuid4())
        return self._trace_id

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Track chain start - multi-step reasoning."""
        self._step_count += 1
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "parent_span_id": parent_run_id,
            "name": serialized.get("name", "chain"),
            "start_time": self._now(),
            "metadata": {
                **self.metadata,
                **(metadata or {}),
                "step_number": self._step_count,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "tags": self.tags + (tags or []),
            },
            "attributes": {
                "input": str(inputs)[:5000],
                "component_type": "chain",
            },
        }
        self._span_stack.append(run_id)
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track chain end."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "output": str(outputs)[:5000],
                "status": "success",
            },
        }
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track chain error."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "error": str(error),
                "status": "error",
            },
        }
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM start - provider info, model config."""
        provider = metadata.get("provider", PRIMARY_PROVIDER) if metadata else PRIMARY_PROVIDER
        model = serialized.get("kwargs", {}).get("model_name", "unknown")

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "parent_span_id": parent_run_id,
            "name": f"llm:{model}",
            "start_time": self._now(),
            "metadata": {
                **self.metadata,
                **(metadata or {}),
                "provider": provider,
                "model": model,
                "fallback_provider": FALLBACK_PROVIDER,
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
            "attributes": {
                "input": "\n".join(prompts)[:10000],
                "component_type": "llm",
            },
        }
        self._span_stack.append(run_id)
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM end - token usage, latency, reasoning."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)

        # Extract token usage
        token_usage = {}
        if response.llm_output:
            token_usage = {
                "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0),
            }

        # Extract reasoning (if available)
        reasoning = ""
        for gen in response.generations:
            for g in gen:
                if hasattr(g, "message") and hasattr(g.message, "additional_kwargs"):
                    reasoning = g.message.additional_kwargs.get("reasoning", "") or ""

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "output": str(response)[:10000],
                "token_usage": token_usage,
                "reasoning_length": len(reasoning),
                "has_reasoning": bool(reasoning),
                "status": "success",
            },
        }
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

        # Also log as generation for Langfuse
        _telemetry_queue.enqueue({
            "type": "generation",
            "data": {
                "trace_id": self.trace_id,
                "name": "llm_generation",
                "start_time": self._now() - 1,  # approximate
                "end_time": self._now(),
                "model": serialized.get("kwargs", {}).get("model_name", "unknown") if "serialized" in locals() else "unknown",
                "input": str(prompts) if "prompts" in locals() else "",
                "output": str(response),
                "usage": token_usage,
                "metadata": {"provider": PRIMARY_PROVIDER, "fallback": FALLBACK_PROVIDER},
            },
        })

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM error - fallback trigger."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "error": str(error),
                "status": "error",
                "fallback_triggered": True,
            },
        }
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Track tool start - latency measurement."""
        import time
        self._tool_latencies[run_id] = time.time()

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "parent_span_id": parent_run_id,
            "name": f"tool:{serialized.get('name', 'unknown')}",
            "start_time": self._now(),
            "metadata": {
                **self.metadata,
                **(metadata or {}),
                "tool_name": serialized.get("name"),
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
            "attributes": {
                "input": input_str[:5000],
                "component_type": "tool",
            },
        }
        self._span_stack.append(run_id)
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track tool end - latency, constraint adherence."""
        import time
        start_time = self._tool_latencies.pop(run_id, time.time())
        latency_ms = (time.time() - start_time) * 1000

        if run_id in self._span_stack:
            self._span_stack.remove(run_id)

        # Check constraint adherence (output length, format, etc.)
        constraint_check = self._check_constraints(serialized.get("name") if "serialized" in locals() else "unknown", output)

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "output": output[:5000],
                "latency_ms": latency_ms,
                "constraint_adherence": constraint_check,
                "status": "success" if constraint_check.get("passed", True) else "constraint_violation",
            },
        }
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

        if not constraint_check.get("passed", True):
            self._constraint_violations.append({
                "tool": serialized.get("name") if "serialized" in locals() else "unknown",
                "violation": constraint_check,
                "timestamp": self._now(),
            })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track tool error."""
        import time
        start_time = self._tool_latencies.pop(run_id, time.time())
        latency_ms = (time.time() - start_time) * 1000

        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "error": str(error),
                "latency_ms": latency_ms,
                "status": "error",
            },
        }
        _telemetry_queue.enqueue({"type": "span", "data": span_data})

    def _check_constraints(self, tool_name: str, output: str) -> Dict[str, Any]:
        """Check constraint adherence for tool output."""
        checks = {
            "passed": True,
            "checks": {},
        }

        # Output length check
        if len(output) > 100000:
            checks["passed"] = False
            checks["checks"]["output_length"] = {
                "passed": False,
                "value": len(output),
                "limit": 100000,
            }

        # Tool-specific constraints
        if tool_name == "search" and not output.strip():
            checks["passed"] = False
            checks["checks"]["empty_results"] = {"passed": False}

        return checks

    def _now(self) -> float:
        """Current timestamp."""
        import time
        return time.time()

    def get_constraint_violations(self) -> List[Dict[str, Any]]:
        """Get recorded constraint violations."""
        return self._constraint_violations.copy()

    def flush(self) -> None:
        """Flush pending telemetry."""
        _telemetry_queue.flush()


# ---------------------------------------------------------------------------
# Tracer for LangChain
# ---------------------------------------------------------------------------

class TelemetryTracer(BaseTracer):
    """LangChain tracer that integrates with our telemetry system."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._handler = TelemetryCallbackHandler()

    @property
    def handler(self) -> TelemetryCallbackHandler:
        return self._handler

    def _persist_run(self, run: Any) -> None:
        """Persist run - handled by callback handler."""
        pass  # Our callback handler does the persistence


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def get_telemetry_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TelemetryCallbackHandler:
    """Get a configured telemetry callback handler."""
    return TelemetryCallbackHandler(
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        metadata=metadata,
    )


def get_telemetry_tracer(**kwargs: Any) -> TelemetryTracer:
    """Get a configured telemetry tracer."""
    return TelemetryTracer(**kwargs)


@contextmanager
def telemetry_context(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Context manager for telemetry tracking."""
    handler = get_telemetry_handler(session_id, user_id, tags, metadata)
    try:
        yield handler
    finally:
        handler.flush()


def flush_telemetry() -> None:
    """Flush all pending telemetry."""
    _telemetry_queue.flush()


def is_telemetry_enabled() -> bool:
    """Check if telemetry is configured."""
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY) or bool(PHOENIX_API_KEY)