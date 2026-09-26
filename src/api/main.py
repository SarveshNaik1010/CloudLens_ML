"""
CloudLens prediction API.

Does not call AWS/Azure/GCP itself -- resource usage is expected to already
be fetched (by CloudLens's own cloud-access helper API) and sent in the
request body. This service's only job: validate the input, run it through
the two trained models, and return predictions as JSON.

Run:
    uvicorn main:app --reload --app-dir src/api

Docs:
    http://localhost:8000/docs
"""
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for model_wrappers
from model_wrappers import CloudLensForecaster, CloudLensOptimizer  # noqa: F401  needed for unpickling

from schemas import PredictRequest, ForecastResponse, OptimizeResponse, AnalyzeResponse
from inference import run_forecast, run_optimize

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'models'
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODELS_DIR / 'cost_forecast_model.pkl', 'rb') as f:
        models['forecaster'] = pickle.load(f)
    with open(MODELS_DIR / 'resource_optimizer_model.pkl', 'rb') as f:
        models['optimizer'] = pickle.load(f)
    yield
    models.clear()


app = FastAPI(
    title="CloudLens Prediction API",
    description="Cost forecasting and resource-optimization scoring for AWS/Azure/GCP usage data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: PredictRequest):
    try:
        return run_forecast(models['forecaster'], req.aws, req.azure, req.gcp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/v1/optimize", response_model=OptimizeResponse)
def optimize(req: PredictRequest):
    try:
        return run_optimize(models['optimizer'], req.aws, req.azure, req.gcp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(req: PredictRequest):
    try:
        fc = run_forecast(models['forecaster'], req.aws, req.azure, req.gcp)
        opt = run_optimize(models['optimizer'], req.aws, req.azure, req.gcp)
        return AnalyzeResponse(forecast=fc, optimization=opt)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
