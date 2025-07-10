from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {'message':'You have hit home'}

@app.get("/about")
def about():
    return {'message':'You have hit about'}
