"""
Cancellation Token Module - AnalystGPT

Provides thread-safe cancellation tokens and exceptions for halting
multi-agent workflows, sandbox code execution, and LLM calls immediately
upon user request without running unnecessary subsequent tasks.
"""

import threading
from typing import Optional, Callable, List, Dict, Any


class WorkflowCancelledException(Exception):
    """Raised when a multi-agent workflow execution is cancelled by the user."""
    def __init__(
        self,
        message: str = "Workflow execution was cancelled by user.",
        task_id: Optional[int] = None,
        step: Optional[str] = None
    ):
        super().__init__(message)
        self.task_id = task_id
        self.step = step
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "cancelled",
            "message": self.message,
            "task_id": self.task_id,
            "step": self.step
        }


class CancellationToken:
    """
    Thread-safe cancellation token shared across WorkflowOrchestrator,
    agents, and sandbox execution loops.
    """
    def __init__(self):
        self._is_cancelled = threading.Event()
        self._reason = ""
        self._cancelled_at_step: Optional[str] = None
        self._task_id: Optional[int] = None
        self._callbacks: List[Callable[[], None]] = []

    def cancel(
        self,
        reason: str = "User requested workflow termination",
        step: Optional[str] = None,
        task_id: Optional[int] = None
    ):
        """Signals cancellation immediately to all executing agents."""
        self._reason = reason
        self._cancelled_at_step = step
        self._task_id = task_id
        self._is_cancelled.set()
        
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    @property
    def is_cancelled(self) -> bool:
        """Returns True if cancellation has been requested."""
        return self._is_cancelled.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def cancelled_at_step(self) -> Optional[str]:
        return self._cancelled_at_step

    @property
    def task_id(self) -> Optional[int]:
        return self._task_id

    def check_cancelled(self, step_name: str = "", task_id: Optional[int] = None):
        """
        Raises WorkflowCancelledException if cancellation has been requested.
        Call this before starting LLM calls, before tool executions, and at turn boundaries.
        """
        if self._is_cancelled.is_set():
            active_step = step_name or self._cancelled_at_step or "active operation"
            active_tid = task_id if task_id is not None else self._task_id
            raise WorkflowCancelledException(
                message=f"Execution halted: {self._reason}",
                task_id=active_tid,
                step=active_step
            )

    def register_callback(self, callback: Callable[[], None]):
        """Registers a callback to be invoked immediately when cancel is called."""
        if self._is_cancelled.is_set():
            try:
                callback()
            except Exception:
                pass
        else:
            self._callbacks.append(callback)
