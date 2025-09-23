from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.datasets import AudioDataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate, DatasetInitRequest
from app.config import BASE_DATA_DIR

from sqlalchemy import or_, and_
from app.models.data_status import  DatasetStatus
from typing import Optional
from datetime import datetime

def get_all_datasets(
    db: Session,
    page: int = 1,
    limit: int = 10,
    status: Optional[str] = None,
    name_search: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
):
    query = db.query(AudioDataset)

    if status:
        query = query.filter(AudioDataset.status == status)

    if name_search:
        query = query.filter(AudioDataset.name.ilike(f"%{name_search}%"))

    if created_from:
        query = query.filter(AudioDataset.created_at >= created_from)

    if created_to:
        query = query.filter(AudioDataset.created_at <= created_to)

    offset = (page - 1) * limit
    total = query.count()
    items = query.order_by(AudioDataset.created_at.desc()).offset(offset).limit(limit).all()

    return {"items": items, "total": total, "page": page, "limit": limit}




def get_dataset_by_id(dataset_id: int, db: Session):
    dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

def create_dataset(dataset: DatasetCreate, db: Session):
    new_dataset = AudioDataset(**dataset.dict())
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    return new_dataset


def update_dataset(dataset_id: int, dataset: DatasetUpdate, db: Session):
    db_dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    for field, value in dataset.dict().items():
        setattr(db_dataset, field, value)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def delete_dataset(dataset_id: int, db: Session):
    db_dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(db_dataset)
    db.commit()



def update_dataset_image(dataset_id: int, dataset_img: str, db: Session):
    db_dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    db_dataset.dataset_img = dataset_img
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def get_dataset_status_by_id(dataset_id: int, db: Session) -> str:
    dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset.status.value  # Если статус — Enum, возвращаем его строковое значение