"""
Configuration for the SmartPack backend.

Everything is read from environment variables (see .env.example) so the
same code runs against a real Postgres instance in production and against
a throwaway SQLite file for quick local development. Postgres is the only
supported database for anything beyond local dev — SQLite is a convenience
fallback, not a target.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Postgres by default. Example:
    #   postgresql+psycopg2://smartpack:smartpack@localhost:5432/smartpack
    # Falls back to a local SQLite file so `uvicorn app.main:app` works with
    # zero setup — swap in real Postgres via DATABASE_URL before deploying.
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///./smartpack_dev.db"
        )
    )

    # Comma-separated list of allowed CORS origins. "*" allows every origin,
    # which is fine for the standalone smartpack.html demo but should be
    # tightened for a real deployment.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "*").split(",")
            if o.strip()
        )
    )

    # Minimum number of real (user-reported) shelf-life feedback rows
    # required before the ML model is trained and used at all. Below this,
    # /recommend simply omits mlPredictedShelfLifeDays rather than training
    # on too few points or inventing a number — see app/ml.py.
    ml_min_training_rows: int = field(
        default_factory=lambda: int(os.getenv("SMARTPACK_ML_MIN_ROWS", "25"))
    )

    # Retrain the ML model at most this often (seconds), since training on
    # every request would be wasteful once feedback volume grows.
    ml_retrain_interval_s: int = field(
        default_factory=lambda: int(os.getenv("SMARTPACK_ML_RETRAIN_S", "300"))
    )

    echo_sql: bool = field(default_factory=lambda: _bool("SMARTPACK_ECHO_SQL", False))


settings = Settings()
