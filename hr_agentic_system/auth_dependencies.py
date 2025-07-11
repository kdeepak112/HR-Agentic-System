from fastapi import Depends, HTTPException, Header
from token_utils import decode_access_token
from sqlalchemy.orm import Session
from db import get_db
from models import User

def get_current_user(authorization: str = Header(..., alias="Authorization"),db: Session = Depends(get_db)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    username = payload["sub"]
    user = db.query(User).filter(User.name == username).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user  # return full user object