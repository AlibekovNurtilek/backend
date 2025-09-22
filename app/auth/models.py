from enum import Enum
from sqlalchemy import Column, Integer, String, Enum as SQLEnum
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
