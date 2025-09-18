from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import timedelta
from fastapi.responses import JSONResponse

from app.auth import models, schemas, utils


def register_user(user: schemas.UserCreate, db: Session) -> schemas.UserOut:
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_pw = utils.hash_password(user.password)
    new_user = models.User(
        username=user.username,
        hashed_password=hashed_pw,
        role='administrator'
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def login_user(credentials: schemas.UserLogin, db: Session) -> JSONResponse:
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
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 60 * 10,     # 10 день
        path="/"
    )
    return response


