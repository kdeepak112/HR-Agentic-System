from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Boolean 
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    role = Column(String)
    password = Column(String)


class employeeChat(Base):
    __tablename__ = 'employee_chats'

    id = Column(Integer, primary_key=True)
    msg_from = Column(Integer, ForeignKey('users.id'))
    msg_content = Column(String)
    date = Column(DateTime, default=datetime.now())
    msg_reply = Column(String)

    user = relationship('User')

class Attendance(Base):
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    date = Column(DateTime, default=datetime.now())
    present = Column(Boolean, default=True)

    user = relationship('User')

class Leave(Base):
    __tablename__ = 'leaves'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    days = Column(Integer)
    reason = Column(String)
    approved = Column(Boolean, default=False)

    user = relationship('User')