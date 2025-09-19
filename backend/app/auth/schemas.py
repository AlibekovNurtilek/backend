from pydantic import BaseModel, constr
from app.auth.models import UserRole

# 🔐 Базовые схемы
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: constr(min_length=4)
    role: UserRole = UserRole.VIEWER

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(UserBase):
    id: int

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

