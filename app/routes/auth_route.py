from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.auth import models, schemas, utils
from datetime import timedelta
from fastapi.responses import JSONResponse
from fastapi import Request
from app.auth.utils import admin_required


router = APIRouter(prefix="/auth", tags=["auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@router.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db), admin = Depends(admin_required)):
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_pw = utils.hash_password(user.password)
    new_user = models.User(
        username=user.username,
        hashed_password=hashed_pw,
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login")
def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == credentials.username).first()
    if not user or not utils.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = utils.create_access_token(
        data={"sub": user.username, "role": user.role},
    )

    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,       # 🚀 защищено от JS (XSS)
        secure=False,        # ⚠️ на prod ставь True (только через HTTPS)
        samesite="lax",      # можно strict, но lax проще на dev
        max_age=60 * 60 * 60 * 10,     # 10 день
        path="/"
    )
    return response

@router.get("/me")
def get_me(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail=f"Not authenticated: {token}")
    
    payload = utils.decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    username = payload.get("username")
    role = payload.get("role")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"username": user.username, "role": role}
    
@router.post("/logout")
def logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("access_token")
    return response

