from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Energy Forecast API", version="1.0")

class InputData(BaseModel):
    values: list[float]  # son 24 saatlik enerji tüketim değerleri

@app.get("/")
def root():
    return {"message": "Energy Forecast API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Basit dummy tahmin: son değeri alıp +0.05 ekle
    prediction = data.values[-1] + 0.05
    return {"prediction": prediction, "input_length": len(data.values)}
