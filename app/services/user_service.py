# user_service.py
from sqlalchemy.orm import Session
from fastapi import HTTPException
from typing import Dict, Any

from app.auth import models, schemas
from app.auth.models import UserRole


def get_all_users(db: Session, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    """
    Возвращает список пользователей с пагинацией.
    """
    if page < 1 or page_size < 1:
        raise HTTPException(status_code=400, detail="Page and page_size must be positive integers")

    total = db.query(models.User).count()
    users = (
        db.query(models.User)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "users": users,
    }


def delete_user(user_id: int, db: Session) -> Dict[str, str]:
    """
    Удаляет пользователя по ID.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role == UserRole.ADMIN.value:
        raise HTTPException(status_code=403, detail="Cannot delete admin user")

    db.delete(user)
    db.commit()
    return {"message": f"User with id {user_id} has been deleted"}
