"""
SmartPack backend.

This API exists to serve one already-written frontend (public/smartpack.html
in the sibling project), which was built expecting this exact contract —
see the "backend wiring" comment near the top of its <script> block. Every
route, field name, and null-vs-omitted choice below is dictated by that
file, not chosen fresh here. If you change a response shape, check what
smartpack.html does with it first (grep for API_BASE).

Endpoints:
    GET   /api/health
    GET   /api/commodities
    GET   /api/materials
    GET   /api/cities
    POST  /api/recommend
    GET   /api/history?limit=50
    PATCH /api/history/{run_id}/feedback
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import engine as reco_engine
from .config import settings
from .database import Base, engine, get_db
from .ml import shelf_life_model
from .models import City, Commodity, Material, RecommendationRun
from .schemas import (
    CityOut,
    CommodityOut,
    FeedbackRequest,
    FeedbackResponse,
    HealthOut,
    HistoryRow,
    MaterialOut,
    MaterialResult,
    RecommendRequest,
    RecommendResponse,
    RequiredOut,
    RouteCityOut,
    RouteOut,
    RouteToOut,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartpack.api")

app = FastAPI(
    title="SmartPack API",
    description="Biology-aware food packaging recommendation backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    # create_all is a safety net for first-run/dev convenience; scripts/seed_db.py
    # is the real migration+seed path and should be run explicitly in any
    # deployment that isn't a scratch dev database.
    Base.metadata.create_all(bind=engine)


# ------------------------------------------------------------------- health --
@app.get("/api/health", response_model=HealthOut)
def health(db: Session = Depends(get_db)) -> HealthOut:
    return HealthOut(
        status="ok",
        mlTrained=shelf_life_model.is_trained,
        mlTrainingRows=shelf_life_model.training_rows,
    )


# ---------------------------------------------------------------- reference --
@app.get("/api/commodities", response_model=list[CommodityOut])
def list_commodities(db: Session = Depends(get_db)) -> list[Commodity]:
    return db.execute(select(Commodity).order_by(Commodity.name)).scalars().all()


@app.get("/api/materials", response_model=list[MaterialOut])
def list_materials(db: Session = Depends(get_db)) -> list[Material]:
    return db.execute(select(Material).order_by(Material.name)).scalars().all()


@app.get("/api/cities", response_model=list[CityOut])
def list_cities(db: Session = Depends(get_db)) -> list[City]:
    return db.execute(select(City).order_by(City.state, City.city)).scalars().all()


# ------------------------------------------------------------------ recommend --
@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, db: Session = Depends(get_db)) -> RecommendResponse:
    commodity = db.execute(
        select(Commodity).where(Commodity.name == payload.commodity)
    ).scalar_one_or_none()
    if commodity is None:
        raise HTTPException(404, f"Unknown commodity: {payload.commodity!r}")

    materials = db.execute(select(Material)).scalars().all()
    if not materials:
        raise HTTPException(503, "No materials seeded — run scripts/seed_db.py")

    temp_c = payload.temp
    life_days = payload.life
    route_out: RouteOut | None = None
    route_from = route_to = None

    if payload.route is not None:
        route_from = db.execute(
            select(City).where(
                City.state == payload.route.fromState, City.city == payload.route.fromCity
            )
        ).scalar_one_or_none()
        route_to = db.execute(
            select(City).where(
                City.state == payload.route.toState, City.city == payload.route.toCity
            )
        ).scalar_one_or_none()
        if route_from is not None and route_to is not None:
            km = reco_engine.haversine_km(
                route_from.lat, route_from.lon, route_to.lat, route_to.lon
            )
            transit_days = max(1, math.ceil(km / 450))
            eff_t = max(temp_c, route_to.peak_temp_c)
            temp_c = eff_t
            life_days = life_days + transit_days
            route_out = RouteOut(
                from_=RouteCityOut(city=route_from.city, state=route_from.state),
                to=RouteToOut(
                    city=route_to.city, state=route_to.state,
                    peakTempC=route_to.peak_temp_c, peakRhPct=route_to.peak_rh_pct,
                    note=route_to.climate_note,
                ),
                km=km, transitDays=transit_days, effectiveTempC=eff_t,
            )
        else:
            route_from = route_to = None

    transport_severity = reco_engine.TRANSPORT_SEVERITY.get(payload.transportMode, 3.5)
    req, results = reco_engine.rank_materials(
        commodity, materials, temp_c, payload.weightG, life_days, payload.rh,
        payload.aw, payload.oxSens, payload.lightSens, payload.priority,
        transport_severity, ph_override=payload.ph,
    )

    # ML: retrain (self-throttled, no-op unless enough real feedback exists
    # and the retrain interval has elapsed) then predict per material.
    shelf_life_model.maybe_retrain(db)
    for r in results:
        r["mlPredictedShelfLifeDays"] = shelf_life_model.predict_one({
            "storage_temp_c": temp_c, "rh_pct": payload.rh, "water_activity": payload.aw,
            "ph": payload.ph, "weight_g": payload.weightG, "target_life_days": life_days,
            "ox_sensitivity": payload.oxSens, "light_sensitivity": payload.lightSens,
            "predicted_shelf_life_days": r["shelfLifeDays"],
            "commodity": payload.commodity, "storage_type": payload.storageType,
            "transport_mode": payload.transportMode, "top_material": r["material"],
        })

    top = results[0] if results else None
    run = RecommendationRun(
        commodity=payload.commodity, commodity_id=commodity.id,
        storage_type=payload.storageType, storage_temp_c=temp_c,
        transport_mode=payload.transportMode, priority=payload.priority,
        weight_g=payload.weightG, rh_pct=payload.rh, water_activity=payload.aw,
        ph=payload.ph, ox_sensitivity=payload.oxSens, light_sensitivity=payload.lightSens,
        target_life_days=life_days,
        route_from_city=route_from.city if route_from else None,
        route_from_state=route_from.state if route_from else None,
        route_to_city=route_to.city if route_to else None,
        route_to_state=route_to.state if route_to else None,
        route_km=route_out.km if route_out else None,
        route_transit_days=route_out.transitDays if route_out else None,
        top_material_id=top["materialId"] if top else None,
        top_material=top["material"] if top else None,
        top_score=top["score"] if top else None,
        predicted_shelf_life_days=top["shelfLifeDays"] if top else None,
        ml_predicted_shelf_life_days=top["mlPredictedShelfLifeDays"] if top else None,
        results_json=results,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return RecommendResponse(
        required=RequiredOut(**req.as_dict()),
        results=[MaterialResult(**r) for r in results],
        route=route_out,
        runId=run.id,
    )


# -------------------------------------------------------------------- history --
@app.get("/api/history", response_model=list[HistoryRow])
def history(limit: int = 50, db: Session = Depends(get_db)) -> list[RecommendationRun]:
    limit = max(1, min(limit, 200))
    return (
        db.execute(
            select(RecommendationRun)
            .order_by(RecommendationRun.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


@app.patch("/api/history/{run_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(
    run_id: str, payload: FeedbackRequest, db: Session = Depends(get_db)
) -> FeedbackResponse:
    run = db.get(RecommendationRun, run_id)
    if run is None:
        raise HTTPException(404, "Unknown run id")
    if run.top_material_id is None or run.top_material_id != payload.materialId:
        raise HTTPException(
            400,
            "Feedback must reference the material this run actually recommended "
            "(top_material_id) — pass the materialId from that run's top result.",
        )

    run.actual_shelf_life_days = payload.actualShelfLifeDays
    run.feedback_recorded_at = datetime.now(timezone.utc)
    db.commit()

    # Real feedback just arrived — let the model pick it up right away
    # rather than waiting for the next periodic retrain window.
    shelf_life_model.maybe_retrain(db, force=True)

    return FeedbackResponse(runId=run.id, actualShelfLifeDays=payload.actualShelfLifeDays)
