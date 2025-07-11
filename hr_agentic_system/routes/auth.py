
from fastapi import APIRouter, Depends, HTTPException
from models import User
from sqlalchemy.orm import Session
from schemas import AuthRequest
from db import get_db
from token_utils import create_access_token
from auth_dependencies import get_current_user

router = APIRouter()

@router.get("/")
def home():
    return {"message": "This is a authentication view."}

@router.post("/authenticate")
def authenticate(auth: AuthRequest,db: Session = Depends(get_db)):

    username = auth.username
    password = auth.password 

    user = db.query(User).filter(User.name == username).first()

    if user:
        if user.password == password:
            token = create_access_token({"sub": username})
            return {"access_token": token, "token_type": "bearer"}
        else:
            raise HTTPException(status_code=401, detail="Invalid password")
    else:
        new_user = User(name=username, password=password, role="user")  # Default role
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        token = create_access_token({"sub": username})
        return {"access_token": token, "token_type": "bearer"}


@router.get("/about")
def about(current_user: str = Depends(get_current_user)):
    return {"message": f"Hello, {current_user}. You are authorized!"}