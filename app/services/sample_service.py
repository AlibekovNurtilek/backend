from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import HTTPException
from app.models.samples import SampleText, ActionType
from app.schemas.sample import SampleCreate, SampleUpdate
from app.models.data_status import SampleStatus
from .action_service import create_action

def get_all_samples(db: Session):
    return db.query(SampleText).all()


def get_sample_by_id(sample_id: int, db: Session):
    sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample

def get_samples_by_dataset_id(
    dataset_id: int,
    db: Session,
    page: int,
    limit: int,
    status: SampleStatus | None = None,
    search: str | None = None,
    from_index: int | None = None,
    to_index: int | None = None
):
    base_query = db.query(SampleText).filter(SampleText.dataset_id == dataset_id)

    # Сортировка по filename (глобальная)
    base_query = base_query.order_by(SampleText.filename.asc())

    # Получаем все id в нужном интервале (это критично!)
    if from_index is not None and to_index is not None:
        limited_ids = (
            base_query
            .with_entities(SampleText.id)
            .offset(from_index)
            .limit(to_index - from_index)
            .all()
        )
        limited_ids = [row.id for row in limited_ids]
        # Ограничим выборку этими id
        filtered_query = db.query(SampleText).filter(SampleText.id.in_(limited_ids))
    else:
        filtered_query = base_query

    # Применяем фильтрацию по статусу
    if status:
        if status == SampleStatus.UNREVIEWED:
            filtered_query = filtered_query.filter(
                SampleText.status.notin_([SampleStatus.APPROVED, SampleStatus.REJECTED])
            )
        else:
            filtered_query = filtered_query.filter(SampleText.status == status)

    # Применяем поиск
    if search:
        pattern = f"%{search.strip()}%"
        filtered_query = filtered_query.filter(
            or_(
                SampleText.filename.ilike(pattern),
                SampleText.text.ilike(pattern)
            )
        )

    total = filtered_query.count()

    offset = (page - 1) * limit
    samples = filtered_query.order_by(SampleText.filename.asc()).offset(offset).limit(limit).all()

    return {
        "dataset_id": dataset_id,
        "page": page,
        "limit": limit,
        "total": total,
        "samples": samples
    }

def create_sample(sample: SampleCreate, db: Session):
    new_sample = SampleText(**sample.dict())
    db.add(new_sample)
    db.commit()
    db.refresh(new_sample)
    return new_sample


def update_sample(sample_id: int, sample: SampleUpdate, db: Session, username: str = None):
    db_sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    db_sample.text = sample.text
    db.commit()
    db.refresh(db_sample)
    if username:
        create_action(
            db=db,
            sample_id=sample_id,
            username=username,
            action_type=ActionType.EDIT,
        )
    return db_sample



def delete_sample(sample_id: int, db: Session):
    db_sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    db.delete(db_sample)
    db.commit()

def approve_sample(sample_id: int, db: Session, username: str = None):
    sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    sample.status = SampleStatus.APPROVED
    db.commit()
    db.refresh(sample)
    if username:
        create_action(
            db=db,
            sample_id=sample_id,
            username=username,
            action_type=ActionType.APPROVE
        )
    return sample

def reject_sample(sample_id: int, db: Session, username: str = None):
    sample = db.query(SampleText).filter(SampleText.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    sample.status = SampleStatus.REJECTED
    db.commit()
    db.refresh(sample)
    if username:
        create_action(
            db=db,
            sample_id=sample_id,
            username=username,
            action_type=ActionType.REJECT
        )
    return sample
