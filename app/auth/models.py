from enum import Enum
from typing import List
from sqlalchemy import Column, Integer, String, Enum as SQLEnum
from pydantic import BaseModel
from app.db import Base

class UserRole(str, Enum):
    ADMIN = "admin"
    ANNOTATOR = "annotator"
    VIEWER = "viewer"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole, name="user_roles"), nullable=False, default=UserRole.VIEWER)


class UserOut(BaseModel):
    id: int
    username: str
    role: UserRole

    class Config:
        orm_mode = True

class UserListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    users: List[UserOut]
