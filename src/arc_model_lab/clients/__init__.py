"""Outbound clients for the external services arc-model-lab consumes.

Each client owns its copy of the provider's wire contract (the request/response
DTOs) rather than importing the provider package, so a contract drift surfaces as
a failing contract test instead of a silent runtime break.
"""

from arc_model_lab.clients.arc_eval_client import (
    ArcEvalClient,
    EvalMetadata,
    EvalMetricResult,
    EvalRequest,
    EvalResponse,
    EvalSettings,
    build_arc_eval_client,
)

__all__ = [
    "ArcEvalClient",
    "EvalMetadata",
    "EvalMetricResult",
    "EvalRequest",
    "EvalResponse",
    "EvalSettings",
    "build_arc_eval_client",
]
