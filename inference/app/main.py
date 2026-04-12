from fastapi import FastAPI
from pydantic import BaseModel
from loader import S3Loader

app = FastAPI()

experiment = {
    'bucket': 'ml-signal-service',
    'directory': 'xg_signal_v1/20260412_162320/'
}

loader = S3Loader(experiment)
model = loader.load_model()
features = loader.load_features()

class PredictRequest(BaseModel):
    tickers: str

@app.post("/predict")
async def predict(req: PredictRequest):
    return
