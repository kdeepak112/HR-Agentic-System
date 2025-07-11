from fastapi import APIRouter, Depends, HTTPException
from db import get_db
from sqlalchemy.orm import Session
from auth_dependencies import get_current_user
from models import employeeChat
from sqlalchemy import desc
from schemas import ChatOut
from typing import List

router = APIRouter()

@router.get("/")
def read_chat():
    return {"message": "This is a get chat route."}


@router.get("/getMessages", response_model=List[ChatOut])
def userChatMessages(current_user: str = Depends(get_current_user) , db: Session = Depends(get_db)):

    chat_obj = db.query(employeeChat).filter(employeeChat.msg_from == current_user.id).order_by(desc(employeeChat.date)).all()

    if not chat_obj:
        raise HTTPException(status_code=404, detail="No messages yet.")

    return chat_obj  # returns list, matches response_model
    