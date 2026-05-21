from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class CountingConfig(BaseModel):
    LABEL: str
    VALUE: float
    PREFIX: str
    THRESHOLD: float | None = None
    THRESHOLD_LABEL: str = ""
    DURATION_SEC: float

    @field_validator("VALUE", mode="before")
    @classmethod
    def coerce_value(cls, v):
        return float(v)

    @field_validator("THRESHOLD", mode="before")
    @classmethod
    def coerce_threshold(cls, v):
        if v is None:
            return None
        return float(v)


class BarItem(BaseModel):
    label: str
    pct: int
    color: str

    @field_validator("pct", mode="before")
    @classmethod
    def coerce_pct(cls, v):
        return int(float(v))


class ComparisonConfig(BaseModel):
    bars_data: list[BarItem]
    UNIT_SUFFIX: str = "%"
    CAPTION: str
    FAIR_LINE_IDX: int
    FAIR_LINE_LABEL: str
    DURATION_SEC: float
    MAX_PCT: int = 0  # auto-computed from bars_data

    @model_validator(mode="after")
    def compute_max_pct(self) -> "ComparisonConfig":
        if self.bars_data:
            self.MAX_PCT = max(b.pct for b in self.bars_data)
        return self


class DeductionItem(BaseModel):
    label: str
    amount: int
    color: str

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v):
        return int(float(v))


class ShrinkConfig(BaseModel):
    GROSS: int
    GROSS_LABEL: str
    deductions: list[DeductionItem]
    DURATION_SEC: float

    @field_validator("GROSS", mode="before")
    @classmethod
    def coerce_gross(cls, v):
        return int(float(v))


class NodeItem(BaseModel):
    title: str
    sub: str
    highlight: bool = False


class FlowConfig(BaseModel):
    nodes_data: list[NodeItem]
    CIRCULAR: bool = False
    RETURN_LABEL: str = ""
    DURATION_SEC: float


class EventItem(BaseModel):
    year: str
    desc: str
    key: bool = False


class TimelineConfig(BaseModel):
    events: list[EventItem]
    year_vals: list[int]
    GAP_LABEL_TEXT: str
    GAP_X_IDX: tuple[int, int]
    DURATION_SEC: float

    @field_validator("year_vals", mode="before")
    @classmethod
    def coerce_year_vals(cls, v):
        return [int(y) for y in v]
