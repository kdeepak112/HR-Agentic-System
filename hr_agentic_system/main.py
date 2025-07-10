from fastapi import FastAPI
from routes import auth, get_messages, chat_messages
from routes.agent_tools import apply_leave, approve_leave, get_attendance, post_attendance, get_reports
from db import Base, engine

app = FastAPI(title="HR Agentic System")
