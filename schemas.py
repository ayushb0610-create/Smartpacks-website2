from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------- reference --
class CommodityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    category: str
    moisture_pct: float
    fat_pct: float
    ph: float | None
    ph_range: str | None = None
    ph_source: str | None = None
    water_activity: float
    respiration_rate: float
    o2_target_pct: float | None
    co2_target_pct: float | None
    base_shelf_life_days: float
    source: str
    usda_temp_note: str | None = None
    usda_rh_note: str | None = None
    usda_life_note: str | None = None


class MaterialOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    bis_standard: str | None = None
    otr: float
    wvtr: float
    thickness_um: float
    cost_index: float
    sustainability: float
    strength: float
    sealability: float
    light_barrier: float
    temp_min_c: float
    temp_max_c: float
    recyclable: str
    note: str
    source: str


class CityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    state: str
    city: str
    lat: float
    lon: float
    peak_temp_c: float
    peak_rh_pct: float
    climate_note: str


# ------------------------------------------------------------- recommend --
class RouteInput(BaseModel):
    fromState: str
    fromCity: str
    toState: str
    toCity: str


class RecommendRequest(BaseModel):
    commodity: str
    storageType: Literal["ambient", "chilled", "frozen"]
    transportMode: Literal["local", "regional", "export"]
    priority: Literal["balanced", "cost", "sustain", "strength"] = "balanced"
    weightG: float = Field(gt=0)
    temp: float
    rh: float = Field(ge=0, le=100)
    life: float = Field(gt=0)
    aw: float = Field(ge=0, le=1)
    oxSens: float = Field(ge=0, le=5)
    lightSens: float = Field(ge=0, le=5)
    ph: float = Field(ge=0, le=14)
    route: RouteInput | None = None


class MaterialResult(BaseModel):
    materialId: int
    material: str
    bisStandard: str | None
    score: int
    otr: float
    wvtr: float
    thickness: float
    sealability: float
    strength: float
    recyclable: str
    note: str
    map: str
    costIndex: float
    costPerM2: str
    packCost: str
    fit: dict[str, int]
    shelfLifeDays: float
    mlPredictedShelfLifeDays: float | None = None


class RequiredOut(BaseModel):
    otr: float
    wvtr: float
    produce: bool
    gas: str
    areaM2: float
    ph: float | None
    resp: float


class RouteCityOut(BaseModel):
    city: str
    state: str


class RouteToOut(RouteCityOut):
    peakTempC: float
    peakRhPct: float
    note: str


class RouteOut(BaseModel):
    from_: RouteCityOut = Field(serialization_alias="from")
    to: RouteToOut
    km: int
    transitDays: int
    effectiveTempC: float

    model_config = ConfigDict(populate_by_name=True)


class RecommendResponse(BaseModel):
    required: RequiredOut
    results: list[MaterialResult]
    route: RouteOut | None
    runId: str


# ------------------------------------------------------------------ history --
class HistoryRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    commodity: str
    storage_type: str
    storage_temp_c: float
    top_material: str | None
    top_score: float | None
    predicted_shelf_life_days: float | None
    ml_predicted_shelf_life_days: float | None
    actual_shelf_life_days: float | None


class FeedbackRequest(BaseModel):
    materialId: int
    actualShelfLifeDays: float = Field(gt=0, le=1000)


class FeedbackResponse(BaseModel):
    ok: bool = True
    runId: str
    actualShelfLifeDays: float


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    mlTrained: bool
    mlTrainingRows: int
