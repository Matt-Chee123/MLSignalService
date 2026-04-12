from fastapi import FastAPI
from pydantic import BaseModel
from inference.loader import S3Loader
from inference.data_preprocess import DataPreprocess

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

@app.post("/predict")
async def predict(req: PredictRequest):
    tickers = req.tickers
    print(tickers)

    data_processor = DataPreprocess(features)
    data = data_processor.prepare(tickers)

    for col in ["fcf", "fcf_sector_z"]:
        print("\n====================")
        print(col)
        print("====================")

        print("dtype:", data[col].dtype)
        print("sample values:")
        print(data[col].dropna().head(10).to_list())

        print("\nunique types inside column:")
        print(data[col].apply(type).value_counts())
    predictions = model.predict(data)
    scored_list = sorted(
        zip(data.index.tolist(), predictions.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_predictions = [
        {
            "ticker": ticker,
            "score": float(score),
            "rank": i + 1
        }
        for i, (ticker, score) in enumerate(scored_list)
    ]

    return {"predictions": ranked_predictions}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ml-signal-api"
    }