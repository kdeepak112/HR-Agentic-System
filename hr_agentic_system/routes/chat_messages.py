from fastapi import APIRouter, Depends, HTTPException
from db import get_db
from sqlalchemy.orm import Session
from auth_dependencies import get_current_user
from models import employeeChat
from sqlalchemy import desc
from schemas import submitChat
from typing import List

router = APIRouter()

@router.get("/")
def read_chat():
    return {"message": "This is a submit chat route."}


@router.post("/postMessages")
def submitUserChatMessages(chat: submitChat,current_user: str = Depends(get_current_user) , db: Session = Depends(get_db),):

    new_chat_obj = employeeChat(msg_content = chat.msg_content , msg_from = current_user.id)  # Default role
    db.add(new_chat_obj)
    db.commit()
    db.refresh(new_chat_obj)
    
    return {'message':'message submitted successfully'}
    