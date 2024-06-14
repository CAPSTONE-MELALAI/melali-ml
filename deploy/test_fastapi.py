from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/random")
async def random_number(): 
    return {"random_number": random.randint(0, 100)}