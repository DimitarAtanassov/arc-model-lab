from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from arc_model_lab.domain import EvaluationError, UnknownMetricError

_EVALUATE_PATH = "/v1/evaluate"

# Wire-contract version this client is written against. arc-eval echoes its own
# version in every response; a mismatch signals the contract has drifted.
CONTRACT_VERSION = "1.0.0"


class EvalMetadata(BaseModel):
    """Caller correlation ids sent to arc-eval. Extra keys are allowed."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    inference_id: str | None = None
    model_id: str | None = None


class EvalRequest(BaseModel):
    """The outbound body for POST /v1/evaluate.

    Mirrors the arc-eval /v1/evaluate contract: metrics, prompt, and
    metadata are all required, and there is no task classification.
    """

    input_text: str
    output_text: str
    prompt: str
    metrics: list[str]
    metadata: EvalMetadata


class EvalMetricResult(BaseModel):
    """One scored metric in an arc-eval response.

    score is intentionally unbounded here: we are liberal in what we accept
    from the provider and let arc-eval own the 0..1 invariant.
    """

    metric_name: str
    score: float
    reasoning: str | None = None
    evaluator_name: str
    evaluator_version: str | None = None


class EvalResponse(BaseModel):
    """The arc-eval response body: only metrics that scored successfully.

    contract_version is optional so an older provider that omits it still
    parses; when present it lets a caller detect provider drift.
    """

    contract_version: str | None = None
    results: list[EvalMetricResult]


class EvalSettings(BaseSettings):
    """Environment-driven configuration for the arc-eval integration.

    Namespaced under ARC_EVAL_ so it composes with the app's ARC_ settings
    without touching them. An empty service_url means evaluation is not wired
    for this environment and requests are skipped rather than failed.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARC_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_url: str = ""
    timeout_seconds: float = 30.0
    # Max concurrent arc-eval calls in a batch fan-out (replay/backfill); bounds
    # outbound load so a large backlog cannot overwhelm arc-eval.
    concurrency: int = Field(default=8, ge=1)


class ArcEvalClient:
    """Asynchronous client for the arc-eval /v1/evaluate endpoint."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    async def evaluate(self, request: EvalRequest) -> EvalResponse:
        """Score one interaction.

        Raises UnknownMetricError when arc-eval does not define a
        requested metric (a 404, a caller error), and EvaluationError
        for every other failure (transport, non-2xx, non-JSON, unexpected schema)
        so the service layer has a single fail-open signal to reason about.
        """
        try:
            response = await self._http.post(_EVALUATE_PATH, json=request.model_dump(mode="json"))
        except httpx.HTTPError as exc:
            raise EvaluationError("arc-eval request failed") from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            raise UnknownMetricError(_error_detail(response) or "requested metric is not defined")

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EvaluationError("arc-eval request failed") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise EvaluationError("arc-eval returned a non-JSON response") from exc

        try:
            return EvalResponse.model_validate(payload)
        except ValidationError as exc:
            raise EvaluationError("arc-eval returned an unexpected schema") from exc

    async def aclose(self) -> None:
        await self._http.aclose()


def build_arc_eval_client(settings: EvalSettings) -> ArcEvalClient | None:
    """Build a client from settings, or None when no service url is configured."""
    if not settings.service_url:
        return None
    http_client = httpx.AsyncClient(base_url=settings.service_url, timeout=settings.timeout_seconds)
    return ArcEvalClient(http_client)


def _error_detail(response: httpx.Response) -> str | None:
    """Best-effort detail string from an arc-eval error body."""
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return None
