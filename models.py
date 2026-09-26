"""
ORM models.

Column names for `commodities`, `materials`, and `cities` are chosen to
match exactly what smartpack.html's mapBackendCommodity / mapBackendMaterial
/ mapBackendCity functions already expect (see public/smartpack.html) — the
frontend was written against this contract before the backend existed, so
the contract is fixed by the frontend, not by us.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Commodity(Base):
    __tablename__ = "commodities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)

    moisture_pct: Mapped[float] = mapped_column(Float, nullable=False)
    fat_pct: Mapped[float] = mapped_column(Float, nullable=False)
    ph: Mapped[float | None] = mapped_column(Float, nullable=True)
    ph_range: Mapped[str | None] = mapped_column(String(60), nullable=True)
    ph_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    water_activity: Mapped[float] = mapped_column(Float, nullable=False)
    respiration_rate: Mapped[float] = mapped_column(Float, nullable=False)
    o2_target_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    co2_target_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_shelf_life_days: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(200), nullable=False)

    usda_temp_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    usda_rh_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    usda_life_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # [[temp_c, resp_rate_ml_per_kg_h], ...] — USDA AH-66 measured points used
    # to interpolate respiration rate at the design temperature (respAt() in
    # the engine). Null for non-produce commodities, which use a flat
    # fat/moisture-derived ceiling instead of a respiration curve.
    respiration_curve: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Material(Base):
    __tablename__ = "materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    bis_standard: Mapped[str | None] = mapped_column(String(120), nullable=True)

    otr: Mapped[float] = mapped_column(Float, nullable=False)  # cc/m2/day
    wvtr: Mapped[float] = mapped_column(Float, nullable=False)  # g/m2/day
    thickness_um: Mapped[float] = mapped_column(Float, nullable=False)
    cost_index: Mapped[float] = mapped_column(Float, nullable=False)  # 1-5
    sustainability: Mapped[float] = mapped_column(Float, nullable=False)  # 1-10
    strength: Mapped[float] = mapped_column(Float, nullable=False)  # 1-10
    sealability: Mapped[float] = mapped_column(Float, nullable=False)  # 1-10
    light_barrier: Mapped[float] = mapped_column(Float, nullable=False)  # 1-10
    temp_min_c: Mapped[float] = mapped_column(Float, nullable=False)
    temp_max_c: Mapped[float] = mapped_column(Float, nullable=False)
    recyclable: Mapped[str] = mapped_column(String(60), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(200), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class City(Base):
    __tablename__ = "cities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state: Mapped[str] = mapped_column(String(80), nullable=False)
    city: Mapped[str] = mapped_column(String(80), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    peak_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    peak_rh_pct: Mapped[float] = mapped_column(Float, nullable=False)
    climate_note: Mapped[str] = mapped_column(String(200), nullable=False)


class RecommendationRun(Base):
    """
    One row per POST /api/recommend call. This is what the "History" tab
    reads (GET /api/history) and what PATCH /api/history/{id}/feedback
    updates once the user reports how long the pack actually lasted — that
    real feedback is the only training signal app/ml.py is allowed to use.
    """

    __tablename__ = "recommendation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Request inputs, snapshotted so history and ML training don't depend on
    # commodities/materials rows never changing underneath them.
    commodity: Mapped[str] = mapped_column(String(120), nullable=False)
    commodity_id: Mapped[int | None] = mapped_column(
        ForeignKey("commodities.id", ondelete="SET NULL"), nullable=True
    )
    storage_type: Mapped[str] = mapped_column(String(20), nullable=False)
    storage_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    transport_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False)
    weight_g: Mapped[float] = mapped_column(Float, nullable=False)
    rh_pct: Mapped[float] = mapped_column(Float, nullable=False)
    water_activity: Mapped[float] = mapped_column(Float, nullable=False)
    ph: Mapped[float] = mapped_column(Float, nullable=False)
    ox_sensitivity: Mapped[float] = mapped_column(Float, nullable=False)
    light_sensitivity: Mapped[float] = mapped_column(Float, nullable=False)
    target_life_days: Mapped[float] = mapped_column(Float, nullable=False)

    route_from_city: Mapped[str | None] = mapped_column(String(80), nullable=True)
    route_from_state: Mapped[str | None] = mapped_column(String(80), nullable=True)
    route_to_city: Mapped[str | None] = mapped_column(String(80), nullable=True)
    route_to_state: Mapped[str | None] = mapped_column(String(80), nullable=True)
    route_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    route_transit_days: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Top-ranked result, denormalised for quick history-table rendering.
    top_material_id: Mapped[int | None] = mapped_column(
        ForeignKey("materials.id", ondelete="SET NULL"), nullable=True
    )
    top_material: Mapped[str | None] = mapped_column(String(120), nullable=True)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_shelf_life_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_predicted_shelf_life_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_shelf_life_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback_recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Full ranked result list (as returned to the client) for reproducibility
    # and as the feature source for ML training — never regenerated or
    # embellished after the fact.
    results_json: Mapped[list] = mapped_column(JSON, nullable=False)

    top_material_ref = relationship("Material", foreign_keys=[top_material_id])
