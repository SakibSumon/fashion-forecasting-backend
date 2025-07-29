# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import load_data, forecast_sku

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

data = load_data()

@app.get("/forecast/{sku}")
def forecast(sku: str, target: str = "revenue", days: int = 30):
    return forecast_sku(data, sku, target, days)

