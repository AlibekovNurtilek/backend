from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime

from app.models.samples import SampleText, SampleAction, ActionType, SampleStatus
from app.auth import models



def create_action(
    db: Session,
    sample_id: int,
    username: int,
    action_type: ActionType,
):
    # 1. Проверяем что sample существует
    sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    # 2. Проверяем что user существует
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 3. Создаём action
    action = SampleAction(
        sample_id=sample.id,
        user_id=user.id,
        action=action_type,
        timestamp=datetime.utcnow(),
    )
    db.add(action)
    db.commit()
    db.refresh(action)

    return action


def get_actions_by_sample(sample_id: int, db: Session):
    sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    return sample.actions


def get_actions_by_user(user_id: int, db: Session):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return db.query(SampleAction).filter(SampleAction.user_id == user_id).all()
