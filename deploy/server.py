from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pipeline import create_recommendation

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/recommendation/")
async def create_recommendation(data: dict):
    idx_selected = data.get("idx_selected")
    budget = data.get("budget")
    days = data.get("days")
    lat_user = data.get("lat_user")
    long_user = data.get("long_user")
    is_accessibility = data.get("is_accessibility")
    
    if not idx_selected or not budget or not days or not lat_user or not long_user:
        raise HTTPException(status_code=400, detail="Bad Request")
    
    recommendation = create_recommendation(idx_selected, budget, days, lat_user, long_user, is_accessibility)
    
    return recommendation