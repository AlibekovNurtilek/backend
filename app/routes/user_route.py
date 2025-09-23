# user_router.py
from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.services import user_service
from app.auth.models import UserListResponse
from app.auth.utils import admin_required


router = APIRouter(prefix="/users", tags=["users"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_model=UserListResponse)
def get_all_users(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(10, ge=1, le=100, description="Размер страницы"),
    db: Session = Depends(get_db),
    admin = Depends(admin_required)
):
    return user_service.get_all_users(db=db, page=page, page_size=page_size)


@router.delete("/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(user_id: int, db: Session = Depends(get_db), admin = Depends(admin_required)):
    return user_service.delete_user(user_id=user_id, db=db)
