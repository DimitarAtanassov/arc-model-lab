"""Application configuration, sourced from environment variables (prefix: ``ARC_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from arc_model_lab.domain.generation import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TEMPERATURE

Device = Literal["auto", "cpu", "mps", "cuda"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARC_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # Database
    database_url: str = "postgresql+psycopg://arc:arc@localhost:5432/arc_model_lab"
    db_echo: bool = False

    # Model registration + loading
    model_name: str = "flan-t5-base-summarizer"
    model_provider: str = "huggingface"
    model_id: str = "google/flan-t5-base"
    model_cache_dir: str | None = None
    tokenizer_id: str = "google/flan-t5-base"
    adapter_path: str | None = None

    # Compute device. "auto" prefers CUDA, then MPS, then CPU. Pin to "cpu" for models
    # too large for the MPS 4 GiB per-tensor limit, which aborts the process otherwise.
    device: Device = "cpu"

    # Generation parameters. These are the server defaults for /inference and smoke
    # runs; they default from the domain constants so "default decoding" has one
    # definition. temperature 0 = greedy/deterministic; > 0 enables sampling.
    max_input_tokens: int = 1024
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    # HTTP server
    api_host: str = "0.0.0.0"  # noqa: S104 — bind all interfaces for containerized serving
    api_port: int = 8000

    # App metadata
    app_name: str = "arc-model-lab"
    environment: str = "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
