from fastapi import FastAPI, HTTPException
import os
app = FastAPI()

MODEL_DIR = '../training/artifacts'

@app.get("/model/predict/{experiment}/{run}")
async def root(experiment, run):
    print("here")
    # file_path = os.path.join(MODEL_DIR, f"{experiment}", f"{run}", 'models/model.pkl')
    # if not os.path.exists(file_path):
    #     raise HTTPException(status_code=404, detail="Config file not found")
    #
    # with open(file_path, "r") as f:
    #     config_data = f.read()
    #     print(f"Training started with: {config_data}")

    return {"status": "training_started", "config": "here"}
