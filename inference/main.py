from fastapi import FastAPI
from pydantic import BaseModel
from inference.loader import S3Loader
from inference.data_preprocess import DataPreprocess
from inference.strategy import StrategyHandler
from datetime import datetime, timezone

app = FastAPI()

experiment = {
    'bucket': 'ml-signal-service',
    'directory': 'xg_signal_v1/20260412_171314/'
}

loader = S3Loader(experiment)
model = loader.load_model()
features = loader.load_features()

class PredictRequest(BaseModel):
    tickers: list[str]
    model_version: str
    top_n: int
    strategy: str

@app.post("/predict")
async def predict(req: PredictRequest):
    tickers = req.tickers
    strategy = req.strategy
    model_version = req.model_version
    top_n = req.top_n
    pred_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    num_tickers = len(tickers)

    data_processor = DataPreprocess(features)
    data = data_processor.prepare(tickers)

    predictions = model.predict(data)

    strategy_processor = StrategyHandler(strategy, top_n)
    constructed_rankings = strategy_processor.construct(tickers, predictions)


    data = {
        "ranked_tickers": constructed_rankings,
        "strategy": strategy,
        "model_version": model_version,
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