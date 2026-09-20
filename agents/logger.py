"""
Centralized Logfire configuration and logging module for AnalystGPT.
Provides structured logging, tracing spans, and integration with Logfire dashboard.
"""

import sys
import logging
from typing import Optional, Any

try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False

_configured = False

def setup_logfire(service_name: str = "analyst-gpt") -> bool:
    """
    Initializes and configures Logfire telemetry.
    Reads credentials from .logfire project config or environment variables.
    """
    global _configured
    if _configured:
        return HAS_LOGFIRE

    if HAS_LOGFIRE:
        try:
            logfire.configure(
                service_name=service_name,
                send_to_logfire="if-token-present",
            )
            # Instrument standard python logging module
            logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])
            _configured = True
            # Mirror every logfire log/span to the local per-run JSONL file.
            # Patch BEFORE the init message so it is captured too.
            from agents.observability import patch_logfire
            patch_logfire()
            logfire.info("Logfire telemetry initialized successfully", service_name=service_name)
        except Exception as e:
            print(f"[Logfire Setup Warning] Failed to configure Logfire: {e}", file=sys.stderr)
    else:
        print("[Logfire Setup Warning] 'logfire' package is not installed.", file=sys.stderr)

    return _configured

# Auto-configure on import
setup_logfire()

def get_logger():
    """Returns logfire if available, otherwise standard logging logger."""
    if HAS_LOGFIRE:
        return logfire
    return logging.getLogger("analyst-gpt")
