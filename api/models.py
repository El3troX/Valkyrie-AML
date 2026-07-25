"""Pydantic request/response models for Valkyrie-AML API."""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class InvestigateRequest(BaseModel):
    query: str
    threshold: float = 0.5
    account_id: Optional[str] = None


class SARRequest(BaseModel):
    account_id: str


class TopAnomaliesRequest(BaseModel):
    n: int = 20
    threshold: float = 0.5


class AccountRequest(BaseModel):
    account_id: str


class AnomalyItem(BaseModel):
    idx: int
    score: float
    sender: str
    receiver: str
    amount: float
    type: str
    risk_level: str
    escalation: str


class DashboardStats(BaseModel):
    total_transactions: int
    flagged_transactions: int
    f1_score: float
    precision: float
    recall: float
    false_positives: int
    optimal_threshold: float
    laundering_typologies: dict[str, int]
    avg_anomaly_score: float
    top_risk_countries: list[dict[str, Any]]


class NetworkNode(BaseModel):
    id: str
    risk_score: float
    pagerank: float
    color: str
    size: float
    label: str


class NetworkEdge(BaseModel):
    source: str
    target: str
    amount: float
    count: int


class NetworkData(BaseModel):
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


class GeoArcData(BaseModel):
    from_country: str
    to_country: str
    from_lat: float
    from_lng: float
    to_lat: float
    to_lng: float
    total_amount: float
    count: int
    max_risk: float
    color: str
