"""
Deterministic packaging-recommendation engine.

This is a line-for-line port of the client-side engine in
public/smartpack.html (functions respAt, required, score, shelfLife,
haversineKm). Keeping the two in lockstep matters: smartpack.html falls
back to its own copy of this math whenever the backend is unreachable, so a
recommendation must come out the same whether it was computed here or in
the browser. If you change a formula, change it in both places and update
CHANGELOG-worthy comments so they don't drift apart silently.

No commodity, material, or scoring constant lives in this file — those
come from the database (see app/seed_data.py for where the numbers
originate and their provenance). This module is pure math over whatever
rows it's handed, and has no external dependencies so it can be unit
tested without a database or a running server.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


def js_round(x: float) -> int:
    """Match JavaScript's Math.round (rounds .5 toward +Infinity), not
    Python's banker's-rounding round()."""
    return math.floor(x + 0.5)


# ---------------------------------------------------------------- weights --
WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {"barrier": 0.4, "cost": 0.15, "sustain": 0.15, "strength": 0.15, "seal": 0.15},
    "cost": {"barrier": 0.3, "cost": 0.4, "sustain": 0.1, "strength": 0.1, "seal": 0.1},
    "sustain": {"barrier": 0.3, "cost": 0.1, "sustain": 0.4, "strength": 0.1, "seal": 0.1},
    "strength": {"barrier": 0.3, "cost": 0.1, "sustain": 0.1, "strength": 0.4, "seal": 0.1},
}

TRANSPORT_SEVERITY: dict[str, float] = {"local": 2, "regional": 3.5, "export": 5}


# --------------------------------------------------------------- protocols --
class CommodityLike(Protocol):
    name: str
    fat_pct: float
    moisture_pct: float
    ph: float | None
    respiration_rate: float
    o2_target_pct: float | None
    co2_target_pct: float | None
    base_shelf_life_days: float
    respiration_curve: list[list[float]] | None


class MaterialLike(Protocol):
    id: int
    name: str
    bis_standard: str | None
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


@dataclass
class Requirement:
    otr: float
    wvtr: float
    produce: bool
    gas: str
    area_m2: float
    ph: float | None
    resp: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "otr": self.otr,
            "wvtr": self.wvtr,
            "produce": self.produce,
            "gas": self.gas,
            "areaM2": self.area_m2,
            "ph": self.ph,
            "resp": self.resp,
        }


# ------------------------------------------------------------------ engine --
def resp_at(commodity: CommodityLike, temp_c: float) -> float:
    """Respiration rate (mL O2/kg·h) at the given temperature, interpolated
    from measured USDA AH-66 points when available, else a Q10≈2.5 scaling
    of the single reference-temperature estimate."""
    curve = commodity.respiration_curve
    if curve and len(curve) > 1:
        pts = sorted(curve, key=lambda p: p[0])
        if temp_c <= pts[0][0]:
            return pts[0][1]
        if temp_c >= pts[-1][0]:
            return pts[-1][1]
        for (t0, r0), (t1, r1) in zip(pts, pts[1:]):
            if t0 <= temp_c <= t1:
                return r0 + (r1 - r0) * (temp_c - t0) / (t1 - t0)
    return commodity.respiration_rate * (2.5 ** ((temp_c - 10) / 10))


def required(
    commodity: CommodityLike,
    temp_c: float,
    weight_g: float,
    life_days: float,
    rh_pct: float,
    water_activity: float,
    ox_sensitivity: float,
) -> Requirement:
    """Target (produce) or maximum-allowed (packaged goods) OTR/WVTR for
    this commodity under the given storage conditions."""
    weight_kg = weight_g / 1000
    area_m2 = 0.15 * max(weight_kg, 0.01) ** (2 / 3)

    rh_gap = max(abs(water_activity * 100 - rh_pct), 5)
    moisture_factor = 50 / rh_gap

    if commodity.respiration_rate > 0:
        rate = resp_at(commodity, temp_c)
        daily_ml = rate * weight_kg * 24
        driving_force = max(0.21 - (commodity.o2_target_pct or 0) / 100, 0.03)
        otr = daily_ml / (area_m2 * driving_force)
        w_loss_budget = 0.05 * weight_g
        wvtr = (w_loss_budget / (area_m2 * max(life_days, 1))) * moisture_factor
        gas = f"O\u2082 {commodity.o2_target_pct}% \u00b7 CO\u2082 {commodity.co2_target_pct}% \u00b7 bal. N\u2082"
        return Requirement(otr, wvtr, True, gas, area_m2, commodity.ph, commodity.respiration_rate)

    ox_factor = max(ox_sensitivity, 0.5) / 3
    otr_max = (60 / (1 + commodity.fat_pct * 1.2)) / ox_factor
    wvtr_base = (
        3 + commodity.moisture_pct
        if commodity.moisture_pct < 10
        else 25 * (commodity.moisture_pct / 100)
    )
    wvtr_max = wvtr_base * moisture_factor
    return Requirement(
        otr_max, wvtr_max, False, "N/A \u2014 flush with N\u2082 or vacuum-seal",
        area_m2, commodity.ph, commodity.respiration_rate,
    )


def score(
    material: MaterialLike,
    req: Requirement,
    temp_c: float,
    priority: str,
    transport_severity: float,
    light_sensitivity: float,
) -> dict[str, Any] | None:
    """Suitability score (0-100) for one material against one requirement,
    or None if the material's usable temperature range excludes temp_c."""
    if material.temp_min_c > temp_c or material.temp_max_c < temp_c:
        return None

    if req.produce:
        ratio = material.otr / req.otr
        barrier = 100 * max(0, 1 - abs(math.log10(max(ratio, 0.001))))
    else:
        otr_fit = (
            100 * (1 - material.otr / req.otr) + 20
            if material.otr <= req.otr
            else max(0, 100 - (material.otr / req.otr - 1) * 40)
        )
        wvtr_fit = (
            100
            if material.wvtr <= req.wvtr
            else max(0, 100 - (material.wvtr / req.wvtr - 1) * 40)
        )
        barrier = min(100, (otr_fit + wvtr_fit) / 2)

    w = WEIGHTS.get(priority, WEIGHTS["balanced"])
    cost_score = 100 * (1 - (material.cost_index - 1) / 4)

    req_strength = 2 + transport_severity * 1.6
    strength_gap = material.strength - req_strength
    strength_score = 100 if strength_gap >= 0 else max(0, 100 + strength_gap * 25)

    total = (
        barrier * w["barrier"]
        + cost_score * w["cost"]
        + material.sustainability * 10 * w["sustain"]
        + strength_score * w["strength"]
        + material.sealability * 10 * w["seal"]
    )

    light_weight = max(0, min(5, light_sensitivity)) / 5 * 0.2
    if light_weight > 0:
        total = total * (1 - light_weight) + (material.light_barrier * 10) * light_weight

    total += 4 if material.bis_standard else -3

    # Indian market cost basis (₹/m²) — see seed_data.py for the ₹110-390/kg
    # band this is scaled across.
    price_per_kg = 110 + (material.cost_index - 1) * 70
    grammage = material.thickness_um * 0.95  # g per m^2
    cost_per_m2 = grammage * price_per_kg / 1000 + 1.2
    pack_cost = cost_per_m2 * req.area_m2

    return {
        "materialId": material.id,
        "material": material.name,
        "bisStandard": material.bis_standard,
        "score": js_round(max(0, min(100, total))),
        "otr": material.otr,
        "wvtr": material.wvtr,
        "thickness": material.thickness_um,
        "sealability": material.sealability,
        "strength": material.strength,
        "recyclable": material.recyclable,
        "note": material.note,
        "map": req.gas if req.produce else "N/A",
        "costIndex": material.cost_index,
        "costPerM2": f"{cost_per_m2:.2f}",
        "packCost": f"{pack_cost:.3f}" if pack_cost < 0.01 else f"{pack_cost:.2f}",
        "fit": {
            "otr": js_round(100 * material.otr / req.otr),
            "wvtr": js_round(100 * material.wvtr / req.wvtr),
        },
    }


def shelf_life_days(
    commodity: CommodityLike, material: MaterialLike, temp_c: float, req: Requirement
) -> float:
    """Deterministic shelf-life estimate (days) for one material, from a
    Q10≈2.5 temperature scaling of the commodity's base shelf life, adjusted
    by how closely the material's barrier matches what's required."""
    q10 = 2.5 ** ((25 - temp_c) / 10)
    life = commodity.base_shelf_life_days * (q10 if commodity.respiration_rate > 0 else 1)
    if req.produce:
        closeness = max(0.3, 1 - abs(math.log10(max(material.otr / req.otr, 0.001))))
    else:
        closeness = 1.2 if material.otr <= req.otr else 0.6
    return max(1, round(life * closeness * 10) / 10)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    r = 6371.0
    to_rad = math.pi / 180
    d_lat = (lat2 - lat1) * to_rad
    d_lon = (lon2 - lon1) * to_rad
    h = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1 * to_rad) * math.cos(lat2 * to_rad) * math.sin(d_lon / 2) ** 2
    )
    return js_round(r * 2 * math.asin(math.sqrt(h)))


def rank_materials(
    commodity: CommodityLike,
    materials: list[MaterialLike],
    temp_c: float,
    weight_g: float,
    life_days: float,
    rh_pct: float,
    water_activity: float,
    ox_sensitivity: float,
    light_sensitivity: float,
    priority: str,
    transport_severity: float,
    ph_override: float | None = None,
) -> tuple[Requirement, list[dict[str, Any]]]:
    """Compute the requirement and every material's score/shelf-life,
    ranked best-first — the full POST /api/recommend response body minus
    routing and history bookkeeping."""
    req = required(commodity, temp_c, weight_g, life_days, rh_pct, water_activity, ox_sensitivity)
    if ph_override is not None:
        req.ph = ph_override

    results: list[dict[str, Any]] = []
    for m in materials:
        s = score(m, req, temp_c, priority, transport_severity, light_sensitivity)
        if s is None:
            continue
        s["shelfLifeDays"] = shelf_life_days(commodity, m, temp_c, req)
        results.append(s)
    results.sort(key=lambda r: r["score"], reverse=True)
    return req, results
