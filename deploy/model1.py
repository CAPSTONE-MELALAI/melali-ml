from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/model1")
async def root():
    return {"message": "model1 masuk"}