from fastapi import FastAPI
from routes import auth, get_messages, chat_messages

app = FastAPI(title="HR Agentic System")

@app.get("/")
def home():
    return {"message": "Welcome to the home page!"}


app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(get_messages.router, prefix="/get_messages", tags=["getMessages"])
app.include_router(chat_messages.router, prefix="/chat_messages", tags=["chatMessages"])