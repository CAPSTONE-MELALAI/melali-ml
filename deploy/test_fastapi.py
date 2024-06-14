from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/random")
async def random_number(): 
    return {"random_number": random.randint(0, 100)}

@app.post("/login")
async def login(data: dict):
    username = data.get("nama")
    password = data.get("password")
    
    # Perform login logic here
    return {"username":username, "password":password}
