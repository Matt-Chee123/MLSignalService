from fastapi import FastAPI
from pydantic import BaseModel
from inference.data_preprocess import DataPreprocess
from inference.strategy import StrategyHandler
from datetime import datetime, timezone
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
import os
from contextlib import asynccontextmanager
import mlflow
import tempfile
import yaml
import numpy as np

MODEL_NAME = os.environ["MODEL_NAME"]
ALIAS = os.environ.get("MODEL_ALIAS", "production")

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = download_artifacts(run_id=mv.run_id, artifact_path="analysis/config.json", dst_path=tmp)
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

    state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    state["features"] = config["features"]
    state["config"] = config
    state["version"] = mv.version
    state["run_id"] = mv.run_id
    print(f"Loaded {MODEL_NAME}@{ALIAS} v{mv.version}, features={state['features']}")
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    tickers: list[str]
    top_n: int
    strategy: str

@app.post("/predict")
async def predict(req: PredictRequest):
    tickers = req.tickers
    strategy = req.strategy
    top_n = req.top_n
    pred_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    num_tickers = len(tickers)

    data_processor = DataPreprocess(state['features'])
    data = data_processor.prepare(tickers)
    float_cols = data.select_dtypes(include=["number"]).columns
    data[float_cols] = data[float_cols].astype(np.float32)
    predictions = state['model'].predict(data)

    strategy_processor = StrategyHandler(strategy, top_n)
    constructed_rankings = strategy_processor.construct(tickers, predictions)


    data = {
        "ranked_tickers": constructed_rankings,
        "strategy": strategy,
        "prediction_timestamp": pred_time,
        "top_n": top_n,
        "tickers_returned": num_tickers
    }

    return data

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ml-signal-api"
    }