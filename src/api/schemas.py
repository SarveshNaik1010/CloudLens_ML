"""
CloudLens API - request/response schemas.

Same shape of resource-usage record for every provider (AWS/Azure/GCP), since
that's what the models were trained on. Fields that aren't reliably available
from a given provider without extra setup (guest-level memory, historical
cost series) are optional -- see README.md "What each provider needs to send"
for exactly what to populate from each provider's own APIs.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator


class ResourceUsageIn(BaseModel):
    resource_id: str = Field(..., description="Provider-native resource identifier (instance ID, VM name, etc.)")
    service: str = Field(..., description="Service name, e.g. 'EC2', 'Virtual Machines', 'Compute Engine'")
    region: str = Field(..., description="Region or zone")
    usage_quantity: float = Field(..., description="Usage amount for the reporting window (e.g. hours running)")
    cost_usd: float = Field(..., description="Actual billed cost for the reporting window, in USD")
    unit_price_usd: Optional[float] = Field(None, description="Price per unit, if known")
    cpu_util_pct: Optional[float] = Field(None, ge=0, le=100, description="Average CPU utilization %")
    mem_util_pct: Optional[float] = Field(None, ge=0, le=100, description="Average memory utilization % (needs a guest agent on all three clouds)")
    net_in_bytes: Optional[float] = Field(None, description="Inbound network bytes over the window")
    net_out_bytes: Optional[float] = Field(None, description="Outbound network bytes over the window")
    duration_hours: Optional[float] = Field(None, description="Hours the resource has been running")
    recent_daily_costs: Optional[List[float]] = Field(
        None, description="Optional last-7-days daily cost history (oldest first). "
                           "Improves forecast accuracy; if omitted the forecast assumes steady state at cost_usd."
    )


class PredictRequest(BaseModel):
    aws: Optional[List[ResourceUsageIn]] = None
    azure: Optional[List[ResourceUsageIn]] = None
    gcp: Optional[List[ResourceUsageIn]] = None

    @model_validator(mode='after')
    def at_least_one_provider(self):
        if not (self.aws or self.azure or self.gcp):
            raise ValueError("Provide resources for at least one of: aws, azure, gcp")
        return self


class ForecastGroup(BaseModel):
    provider: str
    service: str
    region: str
    n_resources: int
    current_cost_usd: float
    predicted_next_period_cost_usd: float
    model_raw_prediction_usd: float = Field(
        ..., description="Unclamped model output, before the low-volume safety clamp described in the README. "
                          "Exposed for transparency/debugging."
    )


class ForecastResponse(BaseModel):
    groups: List[ForecastGroup]
    total_current_cost_usd: float
    total_predicted_next_period_cost_usd: float


class OptimizationTip(BaseModel):
    resource_id: str
    provider: str
    service: str
    region: str
    underutilized_probability: float
    cost_usd: float
    avg_utilization_pct: Optional[float]
    est_monthly_savings_usd: float
    tip: str


class OptimizeResponse(BaseModel):
    n_resources_scanned: int
    n_flagged: int
    total_est_monthly_savings_usd: float
    tips: List[OptimizationTip]


class AnalyzeResponse(BaseModel):
    forecast: ForecastResponse
    optimization: OptimizeResponse
