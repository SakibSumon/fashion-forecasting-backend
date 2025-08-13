
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import load_data, forecast_sku

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

data = load_data()

@app.get("/forecast/{sku}")
def forecast(sku: str, target: str = "RETAIL SALES", months: int = 6):
    valid_targets = ["RETAIL SALES"]
    if target not in valid_targets:
        return {"error": f"Invalid target. Must be one of: {valid_targets}"}
    
    return forecast_sku(data, sku, target, months)
