from pydantic import BaseModel
from datetime import datetime

class AuthRequest(BaseModel):
    username: str
    password: str


class ChatOut(BaseModel):
    id: int
    msg_from: int
    msg_content: str
    date: datetime
    msg_reply : str

    class Config:
        orm_mode = True

class submitChat(BaseModel):
    msg_content: str